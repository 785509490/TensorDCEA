from typing import Callable, Optional
import torch
import math
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import tournament_selection_multifit
from evox.utils import clamp
import time


class TDCEA(Algorithm):
    def __init__(
            self,
            pop_size: int,
            n_objs: int,
            lb: torch.Tensor,
            ub: torch.Tensor,
            max_gen: int = 500,
            taut: int = 100,
            w: float = 0.1,
            istime = False,
            selection_op: Optional[Callable] = None,
            mutation_op: Optional[Callable] = None,
            crossover_op: Optional[Callable] = None,
            device: torch.device | None = None,
    ):
        """Initialize TDCEA algorithm.

        Args:
            pop_size: Population size
            n_objs: Number of objectives
            lb: Lower bounds (1D tensor)
            ub: Upper bounds (1D tensor)
            max_gen: Maximum generations
            taut: Environment change period
            w: Random replacement ratio (default: 0.1)
            selection_op: Selection operator (optional)
            mutation_op: Mutation operator (optional)
            crossover_op: Crossover operator (optional)
            device: Computation device (optional)
        """
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.max_gen = max_gen
        self.taut = taut
        self.istime = istime
        self.w = w
        self.time_step = 0

        if device is None:
            device = torch.get_default_device()

        # Validate bounds
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

        # Initialize two populations
        length = self.ub - self.lb

        # Population 1 (with constraints)
        population = torch.rand(self.pop_size, self.dim, device=device)
        population = length * population + self.lb

        # Population 2 (without constraints)
        population2 = torch.rand(self.pop_size, self.dim, device=device)
        population2 = length * population2 + self.lb

        # Mutable states
        self.pop = Mutable(population)
        self.pop2 = Mutable(population2)
        self.fit = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.fit2 = Mutable(torch.empty((self.pop_size, self.n_objs), device=device).fill_(torch.inf))
        self.fitness1 = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.fitness2 = Mutable(torch.empty(self.pop_size, device=device).fill_(torch.inf))
        self.cons = None

        # Historical populations for prediction (PT array)
        self.PT = [None, None]  # PT[0]: t-2, PT[1]: t-1

        # Archive for feasible solutions
        self.archive = []

        # Old population for change detection
        self.old_pop = None
        self.old_fit = None

    def init_step(self):
        """Initialization step."""
        # Evaluate both populations together
        combined_pop = torch.cat([self.pop, self.pop2], dim=0)
        fitness = self.evaluate(combined_pop)

        if isinstance(fitness, tuple):
            fit, cons = fitness[0], fitness[1]
            total_rows = fit.shape[0]
            mid_point = total_rows // 2

            self.fit = fit[:mid_point]
            self.fit2 = fit[mid_point:]
            self.cons = cons[:mid_point]

            # Environmental selection for Population 1 (with constraints)
            self.pop, self.fit, self.cons, self.fitness1 = self.environmental_selection(
                self.pop, self.fit, self.cons, self.pop_size, True
            )

            # Environmental selection for Population 2 (without constraints)
            self.pop2, self.fit2, self.fitness2 = self.environmental_selection(
                self.pop2, self.fit2, None, self.pop_size, False
            )
        else:
            self.fit = fitness
            self.pop, self.fit, self.fitness1 = self.environmental_selection(
                self.pop, self.fit, None, self.pop_size, False
            )

        # Initialize PT
        self.PT[1] = self.pop.clone()
        self.old_pop = self.pop.clone()
        self.old_fit = self.fit.clone()

    def step(self):
        """Perform one optimization step."""
        # Detect environment change
        is_changed = self.detect_change()

        if is_changed:
            self.time_step += 1
            print(f"Environment change detected at time step {self.time_step}")

            # Save current population to archive
            if self.cons is not None:
                feasible_mask = ~(self.cons > 0).any(dim=1)
                if feasible_mask.any():
                    self.archive.append(self.pop[feasible_mask].clone())

            # Update historical populations
            self.PT[0] = self.PT[1].clone() if self.PT[1] is not None else None
            self.PT[1] = self.pop.clone()

            # Store old population
            self.old_pop = self.pop.clone()

            # Re-evaluate population in new environment
            fitness = self.evaluate(self.pop)
            if isinstance(fitness, tuple):
                self.fit, self.cons = fitness[0], fitness[1]
            else:
                self.fit = fitness
                self.cons = None

            # Detect objective change
            if self.old_fit is not None:
                change_magnitude = torch.abs(self.old_fit - self.fit).sum().item()
                lamda = 1.0 if change_magnitude >= 1e-6 else 0.0
            else:
                lamda = 1.0

            # Calculate feasibility ratio
            if self.cons is not None:
                infeasible_mask = (self.cons > 0).any(dim=1)
                feasible_mask = ~infeasible_mask
                fs_num = feasible_mask.sum().item()
                rp = fs_num / self.pop_size
            else:
                feasible_mask = torch.ones(self.pop_size, dtype=torch.bool, device=self.device)
                infeasible_mask = ~feasible_mask
                fs_num = self.pop_size
                rp = 1.0

            # Generate new population using prediction
            if self.time_step > 1 and self.PT[0] is not None:
                # Calculate center points
                Ct1 = self.PT[0].mean(dim=0)  # Center at t-2
                Ct2 = self.PT[1].mean(dim=0)  # Center at t-1
                delta = Ct2 - Ct1  # Movement direction

                # Generate new solutions for feasible individuals
                if fs_num > 0:
                    old_feasible = self.old_pop[feasible_mask]
                    rand_factors1 = torch.rand(fs_num, self.dim, device=self.device)

                    delta_min = torch.min(delta, torch.zeros_like(delta))
                    delta_max = torch.max(delta, torch.zeros_like(delta))

                    # Dec1: feasible solutions
                    Dec1 = old_feasible + lamda * rp * (
                            delta_min + rand_factors1 * (delta_max - delta_min)
                    )

                    # Boundary repair for Dec1
                    Dec1 = self.boundary_repair(Dec1)
                else:
                    Dec1 = torch.empty(0, self.dim, device=self.device)

                # Generate new solutions for infeasible individuals
                infeas_num = self.pop_size - fs_num
                if infeas_num > 0:
                    old_infeasible = self.old_pop[infeasible_mask]
                    rand_factors2 = torch.rand(infeas_num, self.dim, device=self.device)

                    delta_min = torch.min(delta, torch.zeros_like(delta))
                    delta_max = torch.max(delta, torch.zeros_like(delta))

                    # Dec2: infeasible solutions
                    Dec2 = old_infeasible - (1 - rp) * (
                            delta_min + rand_factors2 * (delta_max - delta_min)
                    )

                    # Boundary repair for Dec2
                    Dec2 = self.boundary_repair(Dec2)
                else:
                    Dec2 = torch.empty(0, self.dim, device=self.device)
            else:
                # First change: no prediction, just keep current individuals
                if fs_num > 0:
                    Dec1 = self.old_pop[feasible_mask]
                else:
                    Dec1 = torch.empty(0, self.dim, device=self.device)

                infeas_num = self.pop_size - fs_num
                if infeas_num > 0:
                    Dec2 = self.old_pop[infeasible_mask]
                else:
                    Dec2 = torch.empty(0, self.dim, device=self.device)

            # Random replacement
            Alpha = round(self.w * self.pop_size)
            NP1 = round(rp * Alpha)
            NP2 = Alpha - NP1

            # Replace in feasible solutions
            if NP1 > 0 and len(Dec1) > 0:
                rand_indices = torch.randperm(len(Dec1), device=self.device)[:NP1]
                rand_solutions = self.lb + (self.ub - self.lb) * torch.rand(
                    NP1, self.dim, device=self.device
                )
                Dec1[rand_indices] = rand_solutions

            # Replace in infeasible solutions
            if NP2 > 0 and len(Dec2) > 0:
                rand_indices = torch.randperm(len(Dec2), device=self.device)[:NP2]
                rand_solutions = self.lb + (self.ub - self.lb) * torch.rand(
                    NP2, self.dim, device=self.device
                )
                Dec2[rand_indices] = rand_solutions

            # Combine Dec1 and Dec2
            if len(Dec1) > 0 and len(Dec2) > 0:
                new_pop = torch.cat([Dec1, Dec2], dim=0)
            elif len(Dec1) > 0:
                new_pop = Dec1
            elif len(Dec2) > 0:
                new_pop = Dec2
            else:
                # Fallback: random initialization
                new_pop = self.lb + (self.ub - self.lb) * torch.rand(
                    self.pop_size, self.dim, device=self.device
                )

            # Ensure correct size
            if len(new_pop) < self.pop_size:
                additional = self.lb + (self.ub - self.lb) * torch.rand(
                    self.pop_size - len(new_pop), self.dim, device=self.device
                )
                new_pop = torch.cat([new_pop, additional], dim=0)
            elif len(new_pop) > self.pop_size:
                new_pop = new_pop[:self.pop_size]

            self.pop = new_pop

            # Evaluate new population
            fitness = self.evaluate(self.pop)
            if isinstance(fitness, tuple):
                self.fit, self.cons = fitness[0], fitness[1]
            else:
                self.fit = fitness
                self.cons = None

            # Environmental selection
            if self.cons is not None:
                self.pop, self.fit, self.cons, self.fitness1 = self.environmental_selection(
                    self.pop, self.fit, self.cons, self.pop_size, True
                )
                # Also update pop2 with same population
                self.pop2, self.fit2, self.fitness2 = self.environmental_selection(
                    self.pop.clone(), self.fit.clone(), None, self.pop_size, False
                )
            else:
                self.pop, self.fit, self.fitness1 = self.environmental_selection(
                    self.pop, self.fit, None, self.pop_size, False
                )

            # Update old population
            self.old_fit = self.fit.clone()

        # Generate offspring
        # Mating pool 1
        mating_pool1 = self.selection(int(self.pop_size / 2), [self.fitness1])
        crossovered1 = self.crossover(self.pop[mating_pool1])
        offspring1 = self.mutation(crossovered1, self.lb, self.ub)
        offspring1 = clamp(offspring1, self.lb, self.ub)

        # Mating pool 2
        mating_pool2 = self.selection(int(self.pop_size / 2), [self.fitness2])
        crossovered2 = self.crossover(self.pop2[mating_pool2])
        offspring2 = self.mutation(crossovered2, self.lb, self.ub)
        offspring2 = clamp(offspring2, self.lb, self.ub)

        # Evaluate offspring
        combined_offspring = torch.cat([offspring1, offspring2], dim=0)
        off_fitness = self.evaluate(combined_offspring)

        if isinstance(off_fitness, tuple):
            off_fit, off_cons = off_fitness[0], off_fitness[1]

            # Merge and select for Population 1 (with constraints)
            merge_pop1 = torch.cat([self.pop, offspring1, offspring2], dim=0)
            merge_fit1 = torch.cat([self.fit, off_fit], dim=0)
            merge_cons1 = torch.cat([self.cons, off_cons], dim=0)

            self.pop, self.fit, self.cons, self.fitness1 = self.environmental_selection(
                merge_pop1, merge_fit1, merge_cons1, self.pop_size, True
            )

            # Merge and select for Population 2 (without constraints)
            merge_pop2 = torch.cat([self.pop2, offspring1, offspring2], dim=0)
            merge_fit2 = torch.cat([self.fit2, off_fit], dim=0)

            self.pop2, self.fit2, self.fitness2 = self.environmental_selection(
                merge_pop2, merge_fit2, None, self.pop_size, False
            )
        else:
            off_fit = off_fitness

            merge_pop1 = torch.cat([self.pop, offspring1, offspring2], dim=0)
            merge_fit1 = torch.cat([self.fit, off_fit], dim=0)

            self.pop, self.fit, self.fitness1 = self.environmental_selection(
                merge_pop1, merge_fit1, None, self.pop_size, False
            )

    def boundary_repair(self, population: torch.Tensor) -> torch.Tensor:
        lb_expand = self.lb.unsqueeze(0).expand_as(population)
        ub_expand = self.ub.unsqueeze(0).expand_as(population)

        # Lower bound violation: reflect
        lb_mask = population < lb_expand
        population = torch.where(
            lb_mask,
            2 * lb_expand - population,
            population
        )

        # Upper bound violation: reflect
        ub_mask = population > ub_expand
        population = torch.where(
            ub_mask,
            2 * ub_expand - population,
            population
        )

        # Final clamp to ensure within bounds
        population = torch.clamp(population, self.lb, self.ub)

        return population

    def environmental_selection(
            self,
            pop: torch.Tensor,
            fit: torch.Tensor,
            cons: torch.Tensor | None,
            n_select: int,
            is_origin: bool
    ) -> tuple:
        from evox.operators.selection import dominate_relation, dominate_relation_cons
        N = fit.shape[0]
        if is_origin and cons is not None:
            Dominate = dominate_relation_cons(fit, fit, cons, cons)
        else:
            Dominate = dominate_relation(fit, fit)
        S = Dominate.sum(dim=1).to(torch.float32)
        R = S @ Dominate.float()
        Distance = torch.cdist(fit, fit, p=2)
        Distance.fill_diagonal_(float('inf'))
        Distance_sorted = torch.sort(Distance, dim=1)[0]

        sqrt_N = int(math.sqrt(N))
        k = min(sqrt_N - 1, Distance_sorted.shape[1] - 1)
        if k >= 0:
            D = 1.0 / (Distance_sorted[:, k] + 2)
        else:
            D = torch.zeros(N, device=self.device)
        Fitness = R + D
        Next = Fitness < 1

        if Next.sum().item() < n_select:
            _, Rank = Fitness.sort()
            Next[Rank[:n_select]] = True
        elif Next.sum().item() > n_select:
            # Truncation
            selected_indices = torch.where(Next)[0]
            selected_fit = fit[selected_indices]
            n_to_remove = Next.sum().item() - n_select
            Del = self.truncation(selected_fit, n_to_remove)
            Next[selected_indices[Del]] = False

        # Extract selected individuals
        selected_pop = pop[Next]
        selected_fit = fit[Next]
        selected_fitness = Fitness[Next]

        if is_origin and cons is not None:
            selected_cons = cons[Next]

        # Sort by fitness
        _, rank = selected_fitness.sort()
        selected_pop = selected_pop[rank]
        selected_fit = selected_fit[rank]
        selected_fitness = selected_fitness[rank]

        if is_origin and cons is not None:
            selected_cons = selected_cons[rank]
            return selected_pop, selected_fit, selected_cons, selected_fitness
        else:
            return selected_pop, selected_fit, selected_fitness

    def truncation(self, pop_obj: torch.Tensor, K: int) -> torch.Tensor:
        Distance = torch.cdist(pop_obj, pop_obj, p=2)
        eye_mask = torch.eye(pop_obj.size(0), dtype=torch.bool, device=self.device)
        Distance[eye_mask] = float('inf')
        Del = torch.zeros(pop_obj.size(0), dtype=torch.bool, device=self.device)
        while Del.sum().item() < K:
            Remain = torch.arange(pop_obj.size(0), device=self.device)[~Del]
            nearest_distances = torch.zeros(len(Remain), device=self.device)
            nearest_indices = torch.zeros(len(Remain), dtype=torch.long, device=self.device)
            for i, ri in enumerate(Remain):
                row = Distance[ri][Remain]
                nearest_distances[i], local_idx = row.min(dim=0)
                nearest_indices[i] = local_idx
            closest_idx = nearest_distances.argmin()
            closest_individual = Remain[nearest_indices[closest_idx]]
            Del[closest_individual] = True
        return Del

    def detect_change(self) -> bool:
        """Detect environment change based on time."""
        if self.istime:
            if self.Dtime > 0 and self.inTime >= self.taut:
                return  True
        else:
            if self.Dtime > 0 and self.Dtime % self.taut == 0:
                return True
        return False


# Utility function for constraint violation calculation
def overall_cv(cv: torch.Tensor) -> torch.Tensor:
    cv = torch.clamp(cv, min=0)
    cv = torch.abs(cv)
    result = cv.sum(dim=1)
    return result