"""
PyTorch-compatible GTGShapley implementation.

This implementation of GTG-Shapley is compatible with PyTorch models and supports both
evaluation mode (for contribution assessment) and differentiable mode (for attacks).
"""

import copy
import math
import numpy as np
import torch
from typing import List, Dict, Tuple, Callable, Union, Optional, Any


class TorchGTGShapley:
    """
    PyTorch-compatible Guided Truncation Gradient (GTG) Shapley Value calculator.
    
    This class implements the GTG-Shapley algorithm using PyTorch tensors, making it
    compatible with both model evaluation and gradient-based optimization.
    
    Key features:
    - Guided permutation sampling to efficiently explore the coalition space
    - Within-round truncation to skip evaluating low-impact coalitions
    - Between-round truncation to skip rounds with minimal improvement
    - Convergence monitoring to reduce sampling when sufficient accuracy is reached
    
    Parameters
    ----------
    num_players : int
        Number of clients/players in the federation
    last_round_utility : float
        Utility value from the previous round (used for comparison)
    eps : float
        Within-round truncation threshold (default: 0.001)
    round_trunc_threshold : float
        Between-round truncation threshold (default: 0.001)
    convergence_criteria : float
        Error tolerance for convergence (default: 0.05)
    last_k : int
        Number of players to check for convergence (default: 10)
    converge_min : int
        Minimum number of permutations to evaluate (default: 30)
    max_percentage : float
        Maximum percentage of permutations to evaluate (default: 0.8)
    prefix_length : int
        Length of the fixed prefix in guided sampling (default: 1)
    normalize : bool
        Whether to normalize Shapley values to sum to 1 (default: True)
    device : torch.device
        Device to use for tensor operations (default: CPU)
    """
    
    def __init__(
        self,
        num_players: int,
        last_round_utility: float = 0.0, # Default to 0.0 instead of inf
        eps: float = 0.001,
        round_trunc_threshold: float = 0.001,
        convergence_criteria: float = 0.05,
        last_k: int = 10,
        converge_min: int = 30,
        max_percentage: float = 0.8,
        prefix_length: int = 1,
        normalize: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        self.num_players = num_players
        self.last_round_utility = last_round_utility
        self.eps = eps  # Within-round truncation threshold
        self.round_trunc_threshold = round_trunc_threshold  # Between-round truncation
        self.convergence_criteria = convergence_criteria  # Convergence error tolerance
        self.last_k = min(last_k, num_players)  # Number of players to check for convergence
        
        # Set minimum number of permutations to evaluate
        self.converge_min = max(converge_min, num_players)
        
        # Calculate maximum number of permutations to evaluate
        self.max_number = min(
            2**num_players,
            max(
                self.converge_min,
                int(max_percentage * (2**num_players)) + np.random.randint(-5, 5)
            ),
        )
        
        self.prefix_length = prefix_length  # Length of fixed prefix in guided sampling
        self.normalize = normalize  # Whether to normalize Shapley values
        self.device = device  # Device to use for tensor operations
        
        # Storage for results
        self.shapley_values = {}
        self.shapley_values_best_subset = {}
        self.utility_function = None
        self.evaluated_subsets = {}  # Cache for evaluated subsets
        
    def set_utility_function(self, utility_function: Callable):
        """
        Set the utility function that evaluates coalition performance.
        
        The utility function should take a list of player indices and return a scalar value.
        
        Parameters
        ----------
        utility_function : callable
            Function that evaluates a coalition's performance
        """
        self.utility_function = utility_function
        
    def evaluate_subset(self, subset: List[int]) -> float:
        """
        Evaluate the utility of a subset of players.
        
        Parameters
        ----------
        subset : List[int]
            Indices of players in the subset
            
        Returns
        -------
        float
            Utility value of the subset
        """
        if not subset:  # Empty subset
            # The utility of the empty coalition should be determined by user's utility_function
            # rather than by using last_round_utility directly.
            # This delegates the empty subset handling to the user-defined function
            # which can consistently return a proper finite value.
            return self.utility_function(subset)
            
        # Convert subset to tuple for caching
        subset_key = tuple(sorted(subset))
        
        # Check if we've already evaluated this subset
        if subset_key in self.evaluated_subsets:
            return self.evaluated_subsets[subset_key]
            
        # Evaluate the subset
        utility = self.utility_function(subset)
        
        # Cache the result
        self.evaluated_subsets[subset_key] = utility
        
        return utility
        
    def compute(self, round_num: int = 0) -> Tuple[Dict[int, float], Dict[int, float]]:
        """
        Compute Shapley values for all players.
        
        Parameters
        ----------
        round_num : int, optional
            Current round number (for logging)
            
        Returns
        -------
        Tuple[Dict[int, float], Dict[int, float]]
            Two dictionaries containing Shapley values:
            1. For all players
            2. For players in the best subset
        """
        # Reset storage
        self.shapley_values = {}
        self.shapley_values_best_subset = {}
        self.evaluated_subsets = {}
        
        # Evaluate complete set
        all_players = list(range(self.num_players))
        current_utility = self.evaluate_subset(all_players)
        
        # Skip calculation if utility didn't improve enough compared to last round
        if abs(current_utility - self.last_round_utility) <= self.round_trunc_threshold:
            # Return zeros for all players if truncated
            shapley_values = {i: 0.0 for i in all_players}
            self.last_round_utility = current_utility
            return shapley_values, shapley_values.copy()
            
        # Initialize storage for convergence tracking
        contribution_records = []
        
        # Initialize index counter
        permutation_index = 0
        
        # Main sampling loop - continue until convergence or max permutations
        while not self._check_convergence(permutation_index, contribution_records):
            # Generate a guided permutation based on the current index
            
            # Determine the starting player for the prefix based on circulation
            start_player = permutation_index % self.num_players
            
            # Create the fixed prefix of length m
            prefix = [(start_player + i) % self.num_players for i in range(self.prefix_length)]
            
            # Identify remaining players
            prefix_set = set(prefix)
            remaining_players = [p for p in all_players if p not in prefix_set]
            
            # Shuffle the remaining players
            np.random.shuffle(remaining_players)
            
            # Combine prefix and shuffled suffix
            perturbed_indices = np.array(prefix + remaining_players, dtype=int)
            
            # Initialize utility values and marginal contributions 
            # Get the empty coalition utility from utility_function rather than using last_round_utility directly
            empty_utility = self.evaluate_subset([])  # This calls utility_function([])
            utilities = [empty_utility]  # v(∅) comes from utility_function
            marginal_contributions = [0.0] * self.num_players
            
            # Process each player in the permutation
            for j in range(self.num_players):
                # Get current player and form subset
                current_player = perturbed_indices[j]
                subset = perturbed_indices[:(j+1)].tolist()
                
                # Check for within-round truncation condition
                # If potential gain is less than eps, stop evaluating this permutation
                if abs(current_utility - utilities[-1]) < self.eps:
                    # Assign 0 marginal contribution to remaining players in this permutation
                    for remaining_player_idx in perturbed_indices[j:]:
                        marginal_contributions[remaining_player_idx] = 0.0
                    # Stop processing this permutation
                    break 
                
                # Evaluate subset (only if not truncated)
                subset_utility = self.evaluate_subset(subset)
                utilities.append(subset_utility)
                
                # Calculate marginal contribution (no clipping)
                marginal_contribution = utilities[-1] - utilities[-2]
                marginal_contributions[current_player] = marginal_contribution
            
            # Store this permutation's results
            contribution_records.append(marginal_contributions)
            
            # Increment permutation index
            permutation_index += 1
                
        # Calculate final Shapley values from all sampled permutations
        shapley_array = np.mean(contribution_records, axis=0)

        # Identify best subset (lowest loss / smallest size)
        best_subset = self._find_best_subset() # Assumes this is now correct

        # Calculate overall marginal gain from last round
        empty_utility = self.evaluate_subset([])
        baseline_utility = self.last_round_utility if self.last_round_utility != float('inf') else empty_utility
        marginal_gain = current_utility - baseline_utility

        # Convert overall shapley array to dictionary
        raw_shapley_values = {i: float(shapley_array[i]) for i in range(self.num_players)}

        # Normalize overall values if requested
        if self.normalize:
            shapley_values = self._normalize_values(raw_shapley_values, marginal_gain)
        else:
            shapley_values = raw_shapley_values

        # Calculate Shapley values for best subset (filtered and potentially normalized)
        # Pass the overall raw values and overall marginal gain
        best_subset_shapley = self._calculate_best_subset_values(
            best_subset, raw_shapley_values, marginal_gain # Pass raw values and overall gain
        )

        # Update last round utility for the *next* round's calculation
        self.last_round_utility = current_utility

        return shapley_values, best_subset_shapley

    def _check_convergence(self, index: int, contribution_records: List) -> bool:
        """Check if sampling has converged."""
        # Always evaluate minimum number of permutations
        if index < self.converge_min:
            return False
            
        # Stop if we've reached the maximum number of permutations
        if index >= self.max_number:
            return True
            
        # If we don't have enough samples, continue
        if len(contribution_records) < self.last_k:
            return False
            
        # Calculate running average of contributions
        all_values = (
            np.cumsum(contribution_records, axis=0) /
            np.reshape(np.arange(1, len(contribution_records) + 1), (-1, 1))
        )
        
        # Take the last_k values for error calculation
        recent_values = all_values[-self.last_k:]
        last_value = all_values[-1:]
        
        # Calculate relative errors
        errors = np.mean(
            np.abs(recent_values - last_value) / 
            (np.abs(last_value) + 1e-12),
            axis=1
        )
        
        # Check if maximum error is below threshold
        return np.max(errors) <= self.convergence_criteria
        
    def _find_best_subset(self) -> List[int]:
        """Find the subset with the lowest utility (loss)."""
        if not self.evaluated_subsets:
            return []

        # Filter out the empty set if it was evaluated
        valid_subsets = {k: v for k, v in self.evaluated_subsets.items() if k}
        if not valid_subsets:
             return []

        # Sort subsets by utility (loss - lower is better) and then by size (smaller is better)
        sorted_subsets = sorted(
            valid_subsets.items(),
            key=lambda x: (x[1], len(x[0])), # Sort by loss ascending, then size ascending
        )

        # Return the best subset (first one after sorting - lowest loss)
        return list(sorted_subsets[0][0]) if sorted_subsets else []

    def _calculate_best_subset_values(
        self, best_subset: List[int], overall_raw_shapley: Dict[int, float], overall_marginal_gain: float
    ) -> Dict[int, float]:
        """Calculate Shapley values for the best subset by filtering overall values.

        Filters the overall Shapley values, setting non-best-subset members to 0.
        Optionally normalizes these filtered values using the overall marginal gain.
        """
        if not best_subset:
            return {i: 0.0 for i in range(self.num_players)}

        best_subset_set = set(best_subset)

        # Create raw values for the best subset by filtering overall values
        raw_values_S = {
            i: (overall_raw_shapley.get(i, 0.0) if i in best_subset_set else 0.0)
            for i in range(self.num_players)
        }

        # Normalize the filtered values if requested, using the overall marginal gain
        if self.normalize:
            # Use the existing normalization function with the filtered values and overall gain
            normalized_values_S = self._normalize_values(raw_values_S, overall_marginal_gain)
            return normalized_values_S # _normalize_values should return all players
        else:
            # Return the filtered, non-normalized values
            return raw_values_S

    def _normalize_values(self, values: Dict[int, float], marginal_gain: float) -> Dict[int, float]:
        """Normalize Shapley values."""
        # Skip normalization if marginal gain is too small
        if abs(marginal_gain) < 1e-10:
            # Instead of returning potentially zero values, use a small uniform value
            # This ensures target client gets some non-zero contribution
            if all(abs(v) < 1e-10 for v in values.values()):
                return {k: 1.0/len(values) for k in values}
            return values
            
        # Separate positive and negative values
        if marginal_gain >= 0:
            sum_value = sum(v for v in values.values() if v >= 0)
            if sum_value < 1e-10:
                # All values are near zero, but we still want to preserve some contribution
                return {k: 1.0/len(values) for k in values}
        else:
            sum_value = sum(v for v in values.values() if v < 0)
            if abs(sum_value) < 1e-10:
                # All values are near zero, but with negative marginal gain
                return {k: -1.0/len(values) for k in values}
                
        # Normalize values to sum to 1 (or -1 for negative marginal gain)
        norm_factor = 1.0 if marginal_gain >= 0 else -1.0
        normalized = {k: norm_factor * v / sum_value for k, v in values.items()}
        
        # Ensure all players are present in the output dict
        normalized = {
            k: norm_factor * values.get(k, 0.0) / sum_value
            for k in range(self.num_players)
        }
        return normalized
            

class DifferentiableTorchGTGShapley(TorchGTGShapley):
    """
    Differentiable version of GTGShapley for gradient-based optimization.
    
    This extension of TorchGTGShapley supports backpropagation through the
    Shapley value computation process, enabling gradient-based attacks.
    
    The key difference is in handling the utility function, which must be
    differentiable with respect to model parameters.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_differentiable(
        self, 
        utility_fn: Callable, 
        params_list: List[List[torch.Tensor]],
        weights: List[float],
        target_client_id: int
    ) -> torch.Tensor:
        """
        Compute Shapley values in a differentiable manner.
        
        This is a simplified version of the compute() method that focuses on
        the target client's contribution and ensures gradient flow.
        
        Parameters
        ----------
        utility_fn : callable
            Differentiable utility function that takes model parameters and returns a scalar
        params_list : List[List[torch.Tensor]]
            List of client parameter lists (each is a list of tensors)
        weights : List[float]
            Client weights for weighted aggregation
        target_client_id : int
            ID of the target client whose contribution we want to minimize
            
        Returns
        -------
        torch.Tensor
            Differentiable Shapley value for the target client
        """
        # Set utility function
        def wrapped_utility_fn(subset):
            # Handle empty subset
            if not subset:
                return self.last_round_utility
                
            # Compute weighted average of parameters for the subset
            subset_weights = [weights[i] for i in subset]
            subset_params = [params_list[i] for i in subset]
            
            # Normalize weights
            total_weight = sum(subset_weights)
            if total_weight > 0:
                norm_weights = [w / total_weight for w in subset_weights]
            else:
                norm_weights = [1.0 / len(subset_weights)] * len(subset_weights)
                
            # Compute weighted average
            avg_params = []
            for param_idx in range(len(subset_params[0])):
                param_tensors = [params[param_idx] for params in subset_params]
                weighted_sum = sum(w * p for w, p in zip(norm_weights, param_tensors))
                avg_params.append(weighted_sum)
                
            # Evaluate utility
            return utility_fn(avg_params)
            
        self.utility_function = wrapped_utility_fn
        
        # We'll use a simplified approach focused on the target client
        # Sample a few permutations that have the target client in different positions
        num_samples = min(20, self.max_number)
        all_players = list(range(self.num_players))
        
        # Get utility of complete set
        full_utility = self.evaluate_subset(all_players)
        
        # Initialize the target's contribution
        target_contribution = torch.tensor(0.0, device=self.device)
        
        for _ in range(num_samples):
            # Create a random permutation
            perm = all_players.copy()
            np.random.shuffle(perm)
            
            # Find the position of the target client
            target_position = perm.index(target_client_id)
            
            # Calculate contribution with and without the target
            with_target = perm[:target_position + 1]
            without_target = perm[:target_position]
            
            # Evaluate both coalitions
            utility_with = self.evaluate_subset(with_target)
            utility_without = self.evaluate_subset(without_target)
            
            # Calculate marginal contribution
            marginal = utility_with - utility_without
            
            # Add to running total
            if isinstance(marginal, torch.Tensor):
                target_contribution = target_contribution + marginal
            else:
                target_contribution = target_contribution + torch.tensor(marginal, device=self.device)
        
        # Average the contributions
        target_contribution = target_contribution / num_samples
        
        return target_contribution