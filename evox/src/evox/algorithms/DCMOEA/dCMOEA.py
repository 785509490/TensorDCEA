from typing import Callable, Optional
import torch
import math
from evox.core import Algorithm, Mutable
from evox.operators.crossover import simulated_binary
from evox.operators.mutation import polynomial_mutation
from evox.operators.selection import nd_environmental_selection_cons, tournament_selection_multifit, \
    nd_environmental_selection
from evox.utils import clamp


class dCMOEA(Algorithm):
    """Dynamic Constrained Multi-objective Evolutionary Algorithm

    Key feature: Uses an archive set A to store historical optimal solutions

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
            selection_op: Optional[Callable] = None,
            mutation_op: Optional[Callable] = None,
            crossover_op: Optional[Callable] = None,
            device: torch.device | None = None,
    ):
        super().__init__()
        self.pop_size = pop_size
        self.n_objs = n_objs
        self.max_gen = max_gen
        self.taut = taut
        self.istime = istime
        self.time_step = 0

        if device is None:
            device = torch.get_default_device()

        assert lb.shape == ub.shape and lb.ndim == 1 and ub.ndim == 1
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
        self.cons = None

        # Archive A for storing non-dominated feasible solutions
        self.archive = None
        self.archive_fit = None
        self.archive_cons = None
        self.calfit = None

        # Store all populations before each change
        self.all_pops = []

    def init_step(self):
        """Initialization step."""
        fitness = self.evaluate(self.pop)
        if isinstance(fitness, tuple):
            self.fit, self.cons = fitness[0], fitness[1]
        else:
            self.fit = fitness
            self.cons = None

        # Initialize archive A
        self.archive, self.archive_fit, self.archive_cons = self.nonselection(
            self.pop, self.fit, self.cons, None, None, None, change=False
        )

    def step(self):
        """Perform one optimization step."""
        # Detect environment change
        is_changed = self.detect_change()

        if is_changed:
            self.time_step += 1
            print(f"Environment change detected at time {self.time_step}")

            # Save feasible solutions before change
            if self.archive is not None and self.archive_cons is not None:
                archive_feasible = ~(self.archive_cons > 0).any(dim=1)
                if archive_feasible.any():
                    self.all_pops.append(self.archive[archive_feasible])
            if self.cons is not None:
                pop_feasible = ~(self.cons > 0).any(dim=1)
                if pop_feasible.any():
                    self.all_pops.append(self.pop[pop_feasible])

            # Re-evaluate archive in new environment
            if self.archive is not None:
                archive_fitness = self.evaluate(self.archive)
                if isinstance(archive_fitness, tuple):
                    self.archive_fit, self.archive_cons = archive_fitness
                else:
                    self.archive_fit = archive_fitness
                    self.archive_cons = None

            # Re-evaluate population in new environment
            pop_fitness = self.evaluate(self.pop)
            if isinstance(pop_fitness, tuple):
                self.fit, self.cons = pop_fitness
            else:
                self.fit = pop_fitness
                self.cons = None

            # React to change
            self.pop, self.fit, self.cons = self.change_response()

            # Update archive with Nonselection2
            self.archive, self.archive_fit, self.archive_cons = self.nonselection2(
                self.pop, self.fit, self.cons,
                self.archive, self.archive_fit, self.archive_cons
            )
        self.calfit = torch.rand(self.pop_size)
        # Generate N offspring one by one
        mating_pool = self.selection(self.pop_size, [self.calfit])
        crossovered = self.crossover(self.pop[mating_pool])
        offspring = self.mutation(crossovered, self.lb, self.ub)
        offspring = clamp(offspring, self.lb, self.ub)

        off_fitness = self.evaluate(offspring)

        # Evaluate offspring
        #off_fitness = self.evaluate(offspring_pop)
        if isinstance(off_fitness, tuple):
            off_fit, off_cons = off_fitness
        else:
            off_fit = off_fitness
            off_cons = None

        # Population selection
        self.pop, self.fit, self.cons = self.population_selection(
            self.pop, self.fit, self.cons,
            offspring, off_fit, off_cons
        )

        # Update archive with Nonselection
        self.archive, self.archive_fit, self.archive_cons = self.nonselection(
            self.pop, self.fit, self.cons,
            self.archive, self.archive_fit, self.archive_cons,
            change=False
        )

    def change_response(self):
        """React to environmental change."""
        n_random = self.pop_size // 2

        # Generate N/2 random individuals
        length = self.ub - self.lb
        random_pop = self.lb + length * torch.rand(n_random, self.dim, device=self.device)

        # Evaluate random population
        random_fitness = self.evaluate(random_pop)
        if isinstance(random_fitness, tuple):
            random_fit, random_cons = random_fitness
        else:
            random_fit = random_fitness
            random_cons = None

        # Calculate feasibility ratio (bili2)
        if random_cons is not None:
            infeasible = (random_cons > 0).any(dim=1)
            feasible_count = (~infeasible).sum().item()
            if feasible_count > 0:
                bili2 = feasible_count / len(random_pop)
            else:
                bili2 = 0.1
        else:
            bili2 = 0.1

        if bili2 == 0:
            bili2 = 0.1

        # Modify objectives for population
        modified_fit = self.modify_objectives(self.pop, self.fit, self.cons)

        # Calculate fitness
        fitness_scores = self.calc_fitness(modified_fit, modified_fit)

        # Select N/2 old solutions (CS) using environmental selection
        cs_indices = self.environmental_selection_by_fitness(
            fitness_scores, modified_fit, n_random
        )
        cs_pop = self.pop[cs_indices]
        cs_fit = self.fit[cs_indices]
        cs_cons = self.cons[cs_indices] if self.cons is not None else None

        # Combine CS and R to get feasible solutions (FS)
        combined_pop = torch.cat([cs_pop, random_pop], dim=0)
        combined_fit = torch.cat([cs_fit, random_fit], dim=0)
        combined_cons = torch.cat([cs_cons, random_cons],
                                  dim=0) if cs_cons is not None and random_cons is not None else None

        if combined_cons is not None:
            feasible_mask = ~(combined_cons > 0).any(dim=1)
            if not feasible_mask.any():
                # No feasible solutions, return combined
                return torch.cat([random_pop, cs_pop], dim=0), \
                    torch.cat([random_fit, cs_fit], dim=0), \
                    torch.cat([random_cons, cs_cons],
                              dim=0) if random_cons is not None and cs_cons is not None else None

            fs_pop = combined_pop[feasible_mask]
            fs_fit = combined_fit[feasible_mask]

            # Get B: individuals with fitness < 2
            fs_fitness = self.calc_fitness(fs_fit, fs_fit)
            b_mask = fs_fitness < 2
            if b_mask.any():
                b_pop = fs_pop[b_mask]

                # Repair infeasible solutions in R
                if random_cons is not None:
                    r_infeasible_mask = (random_cons > 0).any(dim=1)
                    if r_infeasible_mask.any():
                        r_infeasible_pop = random_pop[r_infeasible_mask]
                        repaired_pop = self.repair_infeasible_solutions(
                            r_infeasible_pop, b_pop, bili2
                        )

                        # Replace infeasible R with repaired solutions
                        random_pop = torch.cat([random_pop[~r_infeasible_mask], repaired_pop], dim=0)

                        # Re-evaluate repaired solutions
                        repaired_fitness = self.evaluate(repaired_pop)
                        if isinstance(repaired_fitness, tuple):
                            repaired_fit, repaired_cons = repaired_fitness
                        else:
                            repaired_fit = repaired_fitness
                            repaired_cons = None

                        random_fit = torch.cat([random_fit[~r_infeasible_mask], repaired_fit], dim=0)
                        if random_cons is not None and repaired_cons is not None:
                            random_cons = torch.cat([random_cons[~r_infeasible_mask], repaired_cons], dim=0)

            # Reposition CS individuals
            cs_pop_new = []
            for i in range(len(cs_pop)):
                # Find nearest solution in FS (in objective space)
                distances = torch.norm(cs_fit[i].unsqueeze(0) - fs_fit, dim=1)
                nearest_idx = torch.argmin(distances)
                nearest_individual = fs_pop[nearest_idx]

                # Generate new solution
                new_dec = torch.zeros(self.dim, device=self.device)
                for j in range(self.dim):
                    new_dec[j] = cs_pop[i, j] + torch.rand(1, device=self.device).item() * (
                            nearest_individual[j] - cs_pop[i, j]
                    )

                    # Repair bounds (note: using bili2 instead of rp)
                    if new_dec[j] < self.lb[j]:
                        new_dec[j] = self.lb[j] + torch.rand(1, device=self.device).item() * bili2 * (
                                cs_pop[i, j] - self.lb[j]
                        )
                    elif new_dec[j] > self.ub[j]:
                        new_dec[j] = self.ub[j] - torch.rand(1, device=self.device).item() * bili2 * (
                                cs_pop[i, j] - self.ub[j]
                        )

                cs_pop_new.append(new_dec)

            cs_pop = torch.stack(cs_pop_new)

            # Re-evaluate CS
            cs_fitness = self.evaluate(cs_pop)
            if isinstance(cs_fitness, tuple):
                cs_fit, cs_cons = cs_fitness
            else:
                cs_fit = cs_fitness
                cs_cons = None

        # Combine R and CS as new population
        new_pop = torch.cat([random_pop, cs_pop], dim=0)
        new_fit = torch.cat([random_fit, cs_fit], dim=0)
        new_cons = torch.cat([random_cons, cs_cons], dim=0) if random_cons is not None and cs_cons is not None else None

        return new_pop, new_fit, new_cons

    def repair_infeasible_solutions(self, infeasible_pop, b_pop, bili2):
        """Repair infeasible solutions using feasible solutions in B."""
        repaired_list = []

        for i in range(len(infeasible_pop)):
            # Randomly select a solution from B
            idx = torch.randint(0, len(b_pop), (1,), device=self.device).item()
            if idx == 0 and len(b_pop) > 1:
                idx = 1
            idx = min(idx, len(b_pop) - 1)
            b_individual = b_pop[idx]

            # Generate new solution
            new_dec = torch.zeros(self.dim, device=self.device)
            for k in range(self.dim):
                intermediate = infeasible_pop[i, k] + torch.rand(1, device=self.device).item() * (
                        b_individual[k] - infeasible_pop[i, k]
                )

                # Repair bounds
                if intermediate < self.lb[k]:
                    intermediate = self.lb[k] + torch.rand(1, device=self.device).item() * bili2 * (
                            infeasible_pop[i, k] - self.lb[k]
                    )
                elif intermediate > self.ub[k]:
                    intermediate = self.ub[k] - torch.rand(1, device=self.device).item() * bili2 * (
                            self.ub[k] - intermediate
                    )

                new_dec[k] = intermediate

            repaired_list.append(new_dec)

        return torch.stack(repaired_list)

    def modify_objectives(self, pop, fit, cons):
        """Modify objectives based on constraint violations (vectorized)."""
        if cons is None:
            return fit

        pop_size = fit.shape[0]
        n_objs = fit.shape[1]
        n_cons = cons.shape[1]

        # Calculate infeasible mask and feasible ratio
        infeasible_mask = (cons > 0).any(dim=1)
        rf = (~infeasible_mask).float().mean().item()

        # Get max and min of objectives
        f_max = fit.max(dim=0)[0]
        f_min = fit.min(dim=0)[0]

        # Normalize constraints
        cons_positive = torch.clamp(cons, min=0)
        cons_max = cons.max(dim=0)[0]

        norm_cons = torch.zeros_like(cons_positive)
        valid_cons_mask = cons_max > 0

        if valid_cons_mask.any():
            norm_cons[:, valid_cons_mask] = (
                    cons_positive[:, valid_cons_mask] / cons_max[valid_cons_mask]
            )

        # Average across all constraints
        norm_pop_con = norm_cons.mean(dim=1, keepdim=True)  # [pop_size, 1]

        # Normalize objectives
        f_range = f_max - f_min
        f_range = torch.where(f_range == 0, torch.ones_like(f_range), f_range)
        norm_fit = (fit - f_min) / f_range

        # Calculate Y
        Y = torch.zeros_like(fit)
        Y[infeasible_mask] = norm_fit[infeasible_mask]

        # Calculate modified objectives
        modified_fit = torch.zeros_like(fit)

        if rf == 0:
            # All infeasible: dis = norm_pop_con
            dis = norm_pop_con.expand(-1, n_objs)
            pen = torch.zeros_like(fit)
        else:
            # Calculate distance
            dis = torch.sqrt(norm_fit ** 2 + norm_pop_con ** 2)
            # Calculate penalty
            pen = (1 - rf) * norm_pop_con + rf * Y

        modified_fit = dis + pen

        # Denormalize back
        modified_fit = modified_fit * f_range + f_min

        return modified_fit

    def calc_fitness(self, a_obj, b_obj=None):
        """Calculate fitness based on dominance relations (vectorized).

        Fitness[i] = number of solutions that dominate solution i
        """
        if b_obj is None:
            b_obj = a_obj
            same_pop = True
        else:
            same_pop = torch.equal(a_obj, b_obj)

        n_a = a_obj.shape[0]
        n_b = b_obj.shape[0]

        # Expand for broadcasting: [n_a, 1, n_objs] vs [1, n_b, n_objs]
        a_expanded = a_obj.unsqueeze(1)
        b_expanded = b_obj.unsqueeze(0)

        # Dominance check
        better = (a_expanded < b_expanded).any(dim=2)  # [n_a, n_b]
        worse = (a_expanded > b_expanded).any(dim=2)  # [n_a, n_b]

        # k[i,j] = 1 if a[i] dominates b[j], -1 if b[j] dominates a[i], 0 otherwise
        k = better.long() - worse.long()

        if same_pop:
            # Create upper triangular mask (j > i)
            mask = torch.triu(torch.ones(n_a, n_b, device=a_obj.device, dtype=torch.bool), diagonal=1)

            # For each i: count how many j>i dominate i (k[i,j] == -1)
            dominated_by_later = ((k == -1) & mask).sum(dim=1).float()

            # For each i: count how many j<i dominate i
            # This is equivalent to counting how many i dominate j where j>i
            dominates_later = ((k == 1) & mask).sum(dim=0).float()

            fitness = dominated_by_later + dominates_later
            return fitness
        else:
            # Different populations
            fitness_a = (k == -1).sum(dim=1).float()
            return fitness_a

    def environmental_selection_by_fitness(self, fitness_scores, modified_fit, n_select):
        """Select n_select individuals based on fitness and crowding distance."""
        # Sort by fitness (lower is better)
        sorted_indices = torch.argsort(fitness_scores)

        # If we need exactly n_select, return first n_select
        if len(sorted_indices) <= n_select:
            return sorted_indices

        # Select based on fitness levels
        selected_indices = []
        current_level = 0

        while len(selected_indices) < n_select:
            # Get individuals with current fitness level
            level_mask = fitness_scores == current_level
            level_indices = torch.where(level_mask)[0]

            if len(level_indices) == 0:
                current_level += 1
                continue

            # If adding all individuals at this level doesn't exceed n_select
            if len(selected_indices) + len(level_indices) <= n_select:
                selected_indices.extend(level_indices.tolist())
                current_level += 1
            else:
                # Need to select based on crowding distance
                remaining = n_select - len(selected_indices)
                level_objs = modified_fit[level_indices]
                crowding_dist = self.crowding_distance(level_objs)

                # Sort by crowding distance (descending)
                crowd_sorted = torch.argsort(crowding_dist, descending=True)
                selected_from_level = level_indices[crowd_sorted[:remaining]]
                selected_indices.extend(selected_from_level.tolist())
                break

        return torch.tensor(selected_indices, device=self.device)

    def crowding_distance(self, objectives):
        """Calculate crowding distance for objectives."""
        n = objectives.shape[0]
        m = objectives.shape[1]

        if n <= 2:
            return torch.full((n,), float('inf'), device=self.device)

        crowd_dist = torch.zeros(n, device=self.device)

        for i in range(m):
            # Sort by i-th objective
            sorted_indices = torch.argsort(objectives[:, i])
            sorted_obj = objectives[sorted_indices, i]

            # Boundary points get infinite distance
            crowd_dist[sorted_indices[0]] = float('inf')
            crowd_dist[sorted_indices[-1]] = float('inf')

            # Calculate range
            obj_range = sorted_obj[-1] - sorted_obj[0]
            if obj_range == 0:
                continue

            # Calculate crowding distance for interior points
            for j in range(1, n - 1):
                crowd_dist[sorted_indices[j]] += (sorted_obj[j + 1] - sorted_obj[j - 1]) / obj_range

        return crowd_dist

    def nonselection(self, pop, fit, cons, archive, archive_fit, archive_cons, change=False):
        """Update archive with non-dominated feasible solutions from population."""
        # Handle environment change
        if change and archive is not None and archive_cons is not None:
            feasible_mask = ~(archive_cons > 0).any(dim=1)
            if feasible_mask.any():
                archive = archive[feasible_mask]
                archive_fit = archive_fit[feasible_mask]
                archive_cons = None if archive_cons is None else archive_cons[feasible_mask]

                # Remove dominated solutions
                fitness = self.calc_fitness(archive_fit, archive_fit)
                non_dominated_mask = fitness <= 1
                if non_dominated_mask.any():
                    archive = archive[non_dominated_mask]
                    archive_fit = archive_fit[non_dominated_mask]
                    archive_cons = None if archive_cons is None else archive_cons[non_dominated_mask]
                else:
                    archive = None
                    archive_fit = None
                    archive_cons = None
            else:
                archive = None
                archive_fit = None
                archive_cons = None

        # Get feasible solutions from population
        if cons is not None:
            feasible_mask = ~(cons > 0).any(dim=1)
            if feasible_mask.any():
                feasible_pop = pop[feasible_mask]
                feasible_fit = fit[feasible_mask]

                # Get non-dominated solutions (fitness == 0)
                fitness = self.calc_fitness(feasible_fit, feasible_fit)
                non_dominated_mask = fitness == 0
                if non_dominated_mask.any():
                    ss_pop = feasible_pop[non_dominated_mask]
                    ss_fit = feasible_fit[non_dominated_mask]

                    if archive is None:
                        archive = ss_pop
                        archive_fit = ss_fit
                        archive_cons = None
                    else:
                        # Combine and find non-dominated
                        combined_pop = torch.cat([archive, ss_pop], dim=0)
                        combined_fit = torch.cat([archive_fit, ss_fit], dim=0)

                        fitness = self.calc_fitness(combined_fit, combined_fit)
                        non_dominated_mask = fitness == 0
                        if non_dominated_mask.any():
                            archive = combined_pop[non_dominated_mask]
                            archive_fit = combined_fit[non_dominated_mask]
                            archive_cons = None

                        # Truncate if too large
                        if len(archive) > self.pop_size:
                            truncate_indices = self.environmental_selection_by_fitness(
                                torch.zeros(len(archive), device=self.device),
                                archive_fit,
                                self.pop_size
                            )
                            archive = archive[truncate_indices]
                            archive_fit = archive_fit[truncate_indices]

        return archive, archive_fit, archive_cons

    def nonselection2(self, pop, fit, cons, archive, archive_fit, archive_cons):
        """Update archive by combining with population and keeping non-dominated feasible solutions."""
        # Remove infeasible from archive
        if archive is not None and archive_cons is not None:
            feasible_mask = ~(archive_cons > 0).any(dim=1)
            if feasible_mask.any():
                archive = archive[feasible_mask]
                archive_fit = archive_fit[feasible_mask]
                archive_cons = None
            else:
                archive = None
                archive_fit = None
                archive_cons = None

        # Remove infeasible from population
        if cons is not None:
            feasible_mask = ~(cons > 0).any(dim=1)
            if feasible_mask.any():
                pop = pop[feasible_mask]
                fit = fit[feasible_mask]
            else:
                return archive, archive_fit, archive_cons

        # Combine
        if archive is not None and len(archive) > 0:
            combined_pop = torch.cat([archive, pop], dim=0)
            combined_fit = torch.cat([archive_fit, fit], dim=0)
        else:
            combined_pop = pop
            combined_fit = fit

        if len(combined_pop) > 0:
            # Find non-dominated
            fitness = self.calc_fitness(combined_fit, combined_fit)
            non_dominated_mask = fitness == 0
            if non_dominated_mask.any():
                archive = combined_pop[non_dominated_mask]
                archive_fit = combined_fit[non_dominated_mask]
                archive_cons = None
            else:
                archive = None
                archive_fit = None
                archive_cons = None
        else:
            archive = None
            archive_fit = None
            archive_cons = None

        return archive, archive_fit, archive_cons

    def population_selection(self, pop, fit, cons, offspring, off_fit, off_cons):
        """Select next generation population."""
        # Combine
        combined_pop = torch.cat([pop, offspring], dim=0)
        combined_fit = torch.cat([fit, off_fit], dim=0)
        combined_cons = torch.cat([cons, off_cons], dim=0) if cons is not None and off_cons is not None else None

        if combined_cons is not None:
            infeasible_mask = (combined_cons > 0).any(dim=1)
            feasible_num = (~infeasible_mask).sum().item()

            if feasible_num <= len(combined_pop) // 4:
                # Few feasible solutions (< N/4)
                if feasible_num > 0:
                    feasible_pop = combined_pop[~infeasible_mask]
                    feasible_fit = combined_fit[~infeasible_mask]
                    feasible_cons = torch.zeros(len(feasible_pop), combined_cons.shape[1], device=self.device)
                else:
                    feasible_pop = torch.empty(0, self.dim, device=self.device)
                    feasible_fit = torch.empty(0, self.n_objs, device=self.device)
                    feasible_cons = torch.empty(0, combined_cons.shape[1], device=self.device)

                # Select from infeasible
                if infeasible_mask.any():
                    infeasible_pop = combined_pop[infeasible_mask]
                    infeasible_fit = combined_fit[infeasible_mask]
                    infeasible_cons = combined_cons[infeasible_mask]

                    modified_fit = self.modify_objectives(infeasible_pop, infeasible_fit, infeasible_cons)
                    fitness = self.calc_fitness(infeasible_fit, infeasible_fit)

                    n_select = self.pop_size - len(feasible_pop)
                    if n_select > 0:
                        selected_indices = self.environmental_selection_by_fitness(
                            fitness, modified_fit, n_select
                        )
                        selected_pop = infeasible_pop[selected_indices]
                        selected_fit = infeasible_fit[selected_indices]
                        selected_cons = infeasible_cons[selected_indices]

                        new_pop = torch.cat([selected_pop, feasible_pop], dim=0) if len(
                            feasible_pop) > 0 else selected_pop
                        new_fit = torch.cat([selected_fit, feasible_fit], dim=0) if len(
                            feasible_fit) > 0 else selected_fit
                        new_cons = torch.cat([selected_cons, feasible_cons], dim=0) if len(
                            feasible_cons) > 0 else selected_cons
                    else:
                        new_pop = feasible_pop
                        new_fit = feasible_fit
                        new_cons = feasible_cons
                else:
                    new_pop = feasible_pop
                    new_fit = feasible_fit
                    new_cons = feasible_cons
            else:
                # Many feasible solutions
                modified_fit = self.modify_objectives(combined_pop, combined_fit, combined_cons)
                fitness = self.calc_fitness(combined_fit, combined_fit)

                selected_indices = self.environmental_selection_by_fitness(
                    fitness, modified_fit, self.pop_size
                )
                new_pop = combined_pop[selected_indices]
                new_fit = combined_fit[selected_indices]
                new_cons = combined_cons[selected_indices] if combined_cons is not None else None
        else:
            # No constraints
            new_pop, new_fit, _, _, new_cons = nd_environmental_selection_cons(
                combined_pop, combined_fit, combined_cons, self.pop_size
            ) if combined_cons is not None else (
            *nd_environmental_selection(combined_pop, combined_fit, self.pop_size), None)

        return new_pop, new_fit, new_cons

    def mating_selection(self):
        """Select two parents for mating using tournament selection."""
        # Modify objectives
        modified_fit = self.modify_objectives(self.pop, self.fit, self.cons)

        # Calculate fitness and crowding distance
        fitness = self.calc_fitness(modified_fit, modified_fit)
        crowd_dist = self.crowding_distance(self.fit)

        # Select two parents
        parents = []
        for _ in range(2):
            # Randomly select two candidates
            p1_idx = torch.randint(0, self.pop_size, (1,), device=self.device).item()
            p2_idx = torch.randint(0, self.pop_size, (1,), device=self.device).item()

            # Tournament selection
            if fitness[p1_idx] < fitness[p2_idx]:
                selected = p1_idx
            elif fitness[p1_idx] > fitness[p2_idx]:
                selected = p2_idx
            else:
                # Same fitness, compare crowding distance
                if crowd_dist[p1_idx] > crowd_dist[p2_idx]:
                    selected = p1_idx
                elif crowd_dist[p1_idx] < crowd_dist[p2_idx]:
                    selected = p2_idx
                else:
                    # Random selection
                    selected = p1_idx if torch.rand(1).item() < 0.5 else p2_idx

            parents.append(selected)

        return parents[0], parents[1]

    def detect_change(self) -> bool:
        """Detect environment change based on time."""
        if self.istime:
            if self.Dtime > 0 and self.inTime >= self.taut:
                return  True
        else:
            if self.Dtime > 0 and self.Dtime % self.taut == 0:
                return True
        return False