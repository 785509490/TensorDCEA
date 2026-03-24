from typing import Callable, Optional
import torch
import math
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import nd_environmental_selection_cons, tournament_selection_multifit, \
    nd_environmental_selection
from evox.utils import clamp


class DC_MOEA(Algorithm):
    """Dynamic Constrained Multi-objective Evolutionary Algorithm

    Reference:
    K. Deb, U. Bhaskara Rao N., and S. Karthik, Dynamic multi-objective
    optimization and decision-making using modified NSGA-II: A case study on
    hydro-thermal power scheduling, EMO 2007, 803-817.
    """

    def __init__(
            self,
            pop_size: int,
            n_objs: int,
            lb: torch.Tensor,
            ub: torch.Tensor,
            max_gen: int = 500,
            taut: int = 100,
            istime=False,
            reinit_type: int = 2,  # 1: Mutation based, 2: Random reinitialization
            zeta: float = 0.2,  # Ratio of reinitialized solutions
            selection_op: Optional[Callable] = None,
            mutation_op: Optional[Callable] = None,
            crossover_op: Optional[Callable] = None,
            device: torch.device | None = None,
    ):
        """
        :param pop_size: The size of the population.
        :param n_objs: The number of objective functions.
        :param lb: Lower bounds for decision variables (1D tensor).
        :param ub: Upper bounds for decision variables (1D tensor).
        :param max_gen: Maximum number of generations.
        :param taut: Change detection period.
        :param reinit_type: 1 for mutation-based, 2 for random reinitialization.
        :param zeta: Ratio of reinitialized solutions.
        :param device: Computation device.
        """
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.max_gen = max_gen
        self.taut = taut
        self.istime = istime
        self.reinit_type = reinit_type
        self.zeta = zeta
        self.gen_count = 0
        self.time_step = 0
        self.flag = False

        if device is None:
            device = torch.get_default_device()

        # Check bounds
        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
        assert lb.dtype == ub.dtype and lb.device == ub.device
        self.dim = lb.shape[0]

        self.lb = lb.to(device=device)
        self.ub = ub.to(device=device)
        self.device = device

        # Set operators
        self.selection = selection_op if selection_op is not None else tournament_selection_multifit
        self.mutation = mutation_op if mutation_op is not None else polynomial_mutation
        self.crossover = crossover_op if crossover_op is not None else simulated_binary

        # Initialize population
        length = self.ub - self.lb
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb

        self.pop = Mutable(population)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.rank = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.dis = Mutable(torch.empty(self.pop_size, device=device).fill_(-torch.inf))
        self.cons = None

        # Archive for storing populations before each change
        self.all_pops = []

    def init_step(self):
        """Initialization step."""
        fitness = self.evaluate(self.pop)
        if isinstance(fitness, tuple):
            self.fit = fitness[0]
            self.cons = fitness[1]
            _, _, self.rank, self.dis, self.cons = nd_environmental_selection_cons(
                self.pop, self.fit, self.cons, self.pop_size
            )
        else:
            self.fit = fitness
            _, _, self.rank, self.dis = nd_environmental_selection(
                self.pop, self.fit, self.pop_size
            )

    def step(self):
            # Detect change
            is_changed = self.detect_change()

            if is_changed:
                self.time_step += 1

                # Save feasible population before change
                if self.cons is not None:
                    feasible_mask = ~(self.cons > 0).any(dim=1)
                    feasible_pop = self.pop[feasible_mask]
                    if len(feasible_pop) > 0:
                        self.all_pops.append(feasible_pop)

                # Reinitialization
                self.reinitialize_population()

                # Re-evaluate in new environment
                fitness = self.evaluate(self.pop)
                if isinstance(fitness, tuple):
                    self.fit, self.cons = fitness
                else:
                    self.fit = fitness
                    self.cons = None

                # Handle infeasible solutions
                if self.cons is not None:
                    self.handle_infeasible_solutions()

            # Normal evolution with modified objective
            mating_pool = self.selection(self.pop_size, [-self.dis, self.rank])
            crossovered = self.crossover(self.pop[mating_pool])
            offspring = self.mutation(crossovered, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)

            off_fit = self.evaluate(offspring)

            if isinstance(off_fit, tuple):
                off_fit, off_cons = off_fit
                merge_cons = torch.cat([self.cons, off_cons], dim=0)
                merge_pop = torch.cat([self.pop, offspring], dim=0)
                merge_fit = torch.cat([self.fit, off_fit], dim=0)

                # Modify objectives based on constraints
                modified_fit = self.modify_objectives(merge_pop, merge_fit, merge_cons)

                self.pop, self.fit, self.rank, self.dis, self.cons = nd_environmental_selection_cons(
                    merge_pop, modified_fit, merge_cons, self.pop_size
                )
            else:
                merge_pop = torch.cat([self.pop, offspring], dim=0)
                merge_fit = torch.cat([self.fit, off_fit], dim=0)
                self.pop, self.fit, self.rank, self.dis = nd_environmental_selection(
                    merge_pop, merge_fit, self.pop_size
                )

            self.gen_count += 1

    def reinitialize_population(self):
        """Reinitialize a portion of the population."""
        n_reinit = math.ceil(self.pop_size * self.zeta)
        if n_reinit <= 0:
            return

        indices = torch.randperm(self.pop_size, device=self.device)[:n_reinit]

        if self.reinit_type == 1:
            # Mutation-based reinitialization
            selected_pop = self.pop[indices]
            mutated = self.mutation(selected_pop, self.lb, self.ub)
            self.pop[indices] = clamp(mutated, self.lb, self.ub)
        else:
            # Random reinitialization
            length = self.ub - self.lb
            new_decs = self.lb + length * torch.rand(n_reinit, self.dim, device=self.device)
            self.pop[indices] = new_decs

    def handle_infeasible_solutions(self):
        """Handle infeasible solutions by mating with nearest feasible solutions."""
        if self.cons is None:
            return

        # Separate feasible and infeasible solutions
        infeasible_mask = (self.cons > 0).any(dim=1)
        feasible_mask = ~infeasible_mask

        infeasible_pop = self.pop[infeasible_mask]
        feasible_pop = self.pop[feasible_mask]

        if len(infeasible_pop) == 0 or len(feasible_pop) == 0:
            return

        # For each infeasible solution, find nearest feasible solution
        repaired_pop = []

        for i in range(len(infeasible_pop)):
            # Calculate distance in decision space
            distances = torch.norm(
                infeasible_pop[i].unsqueeze(0) - feasible_pop,
                dim=1
            )
            nearest_idx = torch.argmin(distances)

            # Create offspring from infeasible and nearest feasible
            parents = torch.stack([infeasible_pop[i], feasible_pop[nearest_idx]])
            offspring_cross = self.crossover(parents)
            offspring = self.mutation(offspring_cross, self.lb, self.ub)
            offspring = clamp(offspring, self.lb, self.ub)

            # Evaluate candidates
            candidates = torch.cat([
                infeasible_pop[i].unsqueeze(0),
                feasible_pop[nearest_idx].unsqueeze(0),
                offspring
            ], dim=0)

            cand_fitness = self.evaluate(candidates)
            if isinstance(cand_fitness, tuple):
                cand_cons = cand_fitness[1]
                # Select the one with minimum constraint violation
                cons_sum = cand_cons.sum(dim=1)
                best_idx = torch.argmin(cons_sum)
                repaired_pop.append(candidates[best_idx])

        if len(repaired_pop) > 0:
            repaired_tensor = torch.stack(repaired_pop)
            # Add repaired solutions to population
            self.pop = torch.cat([self.pop, repaired_tensor], dim=0)

            # Re-evaluate
            rep_fitness = self.evaluate(repaired_tensor)
            if isinstance(rep_fitness, tuple):
                rep_fit, rep_cons = rep_fitness
                self.fit = torch.cat([self.fit, rep_fit], dim=0)
                self.cons = torch.cat([self.cons, rep_cons], dim=0)
            else:
                self.fit = torch.cat([self.fit, rep_fitness], dim=0)

    def modify_objectives(self, pop, fit, cons):
        """Modify objectives based on constraint violations.

        Args:
            pop: Population decision variables [pop_size, dim]
            fit: Objective values [pop_size, n_objs]
            cons: Constraint violations [pop_size, n_cons]

        Returns:
            modified_fit: Modified objective values [pop_size, n_objs]
        """
        if cons is None:
            return fit

        # Calculate infeasible mask (any constraint violation > 0)
        infeasible_mask = (cons > 0).any(dim=1)  # [pop_size]
        rf = (~infeasible_mask).float().mean().item()  # Feasible ratio

        pop_size = fit.shape[0]
        n_objs = fit.shape[1]
        n_cons = cons.shape[1]

        # Get max and min of objectives across population
        f_max = fit.max(dim=0)[0]  # [n_objs]
        f_min = fit.min(dim=0)[0]  # [n_objs]

        # Clamp negative violations to 0
        cons_positive = torch.clamp(cons, min=0)  # [pop_size, n_cons]

        # For each constraint, get the maximum violation across population
        cons_max = cons.max(dim=0)[0]  # [n_cons]

        # Normalize constraint violations (vectorized implementation)
        # Initialize normalized constraints
        norm_cons = torch.zeros_like(cons_positive)  # [pop_size, n_cons]

        # Create mask for constraints with max violation > 0
        valid_cons_mask = cons_max > 0  # [n_cons]

        # Normalize only valid constraints
        if valid_cons_mask.any():
            # Broadcasting: cons_positive[:, valid_cons_mask] / cons_max[valid_cons_mask]
            norm_cons[:, valid_cons_mask] = (
                    cons_positive[:, valid_cons_mask] / cons_max[valid_cons_mask]
            )

        # Calculate normalized constraint violation for each individual
        # Average across all constraints: (1/n_cons) * sum
        norm_pop_con = norm_cons.mean(dim=1, keepdim=True)  # [pop_size, 1]

        # Normalize objectives
        f_range = f_max - f_min
        # Avoid division by zero
        f_range = torch.where(f_range == 0, torch.ones_like(f_range), f_range)
        norm_fit = (fit - f_min) / f_range  # [pop_size, n_objs]

        # Modify objectives based on constraint violations and feasible ratio
        # The modification depends on whether solution is feasible or not
        modified_fit = torch.zeros_like(fit)

        if rf > 0:  # If there are feasible solutions
            # For feasible solutions: use original normalized objectives
            modified_fit[~infeasible_mask] = norm_fit[~infeasible_mask]

            # For infeasible solutions: add penalty based on constraint violation
            # Use (1 + norm_pop_con) to make infeasible solutions worse
            modified_fit[infeasible_mask] = (
                    norm_fit[infeasible_mask] + norm_pop_con[infeasible_mask]
            )
        else:  # If no feasible solutions exist
            # All solutions are infeasible, rank by constraint violation
            # Lower constraint violation is better
            modified_fit = norm_fit + norm_pop_con

        # Denormalize back to original scale
        modified_fit = modified_fit * f_range + f_min

        return modified_fit

    def detect_change(self) -> bool:
        """Detect environment change based on time."""
        if self.istime:
            if self.Dtime > 0 and self.inTime >= self.taut:
                return  True
        else:
            if self.Dtime > 0 and self.Dtime % self.taut == 0:
                return True
        return False