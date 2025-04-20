import numpy as np
import pytorch_lightning as pl
import torch

from ctd.comparison.utils import FixedPoints


def find_fixed_points(
    model: pl.LightningModule,
    state_trajs: np.array,
    inputs: np.array,
    n_inits=1024,
    noise_scale=0.2,
    learning_rate=1e-2,
    max_iters=10000,
    device="cpu",
    seed=0,
    compute_jacobians=False,
    n_restarts=3,
    temperature=1.0,
    q_threshold=1e-6,
    early_stop_threshold=1e-8,
):
    """
    Improved fixed point finder with multiple restarts and simulated annealing.
    
    Args:
        model: The model to find fixed points for
        state_trajs: Trajectory of states 
        inputs: The inputs to the model
        n_inits: Number of initial points to sample
        noise_scale: Scale of noise to add to initial states
        learning_rate: Learning rate for optimization
        max_iters: Maximum number of iterations
        device: Device to run on
        seed: Random seed
        compute_jacobians: Whether to compute Jacobians for stability analysis
        n_restarts: Number of optimization restarts with different initializations
        temperature: Initial temperature for simulated annealing
        q_threshold: Threshold for considering a point as fixed
        early_stop_threshold: Threshold for early stopping
    
    Returns:
        all_fps: FixedPoints object containing the fixed points and related data
    """
    # Set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
    state_trajs = torch.tensor(state_trajs, device=device, dtype=torch.float32)

    model = model.to(device)
    state_trajs = state_trajs.to(device)
    inputs = inputs.to(device)

    # Prevent gradient computation for the neural ODE
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Choose random points along the observed trajectories
    if len(state_trajs.shape) > 2:
        n_samples, n_steps, state_dim = state_trajs.shape
        state_pts = state_trajs.reshape(-1, state_dim)
        if len(inputs.shape) > 1:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        idx = torch.randint(n_samples * n_steps, size=(n_inits,), device=device)
    else:
        n_samples_steps, state_dim = state_trajs.shape
        state_pts = state_trajs
        idx = torch.randint(n_samples_steps, size=(n_inits,), device=device)

    # Select the initial states
    states = state_pts[idx]
    if len(inputs.shape) > 1:
        inputs = inputs[idx]
    else:
        inputs = inputs.unsqueeze(0).repeat(n_inits, 1)

    # Create a larger pool of initial states for multiple restarts
    expanded_states_list = []
    expanded_inputs_list = []
    
    for i in range(n_restarts):
        # Add different levels of noise for each restart
        noise_level = noise_scale * (1.0 + 0.5 * i)
        noisy_states = states.clone().detach() + noise_level * torch.randn_like(states, device=device)
        expanded_states_list.append(noisy_states)
        expanded_inputs_list.append(inputs.clone())
    
    # Keep as separate tensors to avoid the non-leaf tensor error
    initial_states_full = []
    for states_batch in expanded_states_list:
        initial_states_full.append(states_batch.cpu().numpy())
    initial_states = np.concatenate(initial_states_full, axis=0)
    
    # Make each tensor require gradients separately
    for i in range(len(expanded_states_list)):
        expanded_states_list[i].requires_grad = True
    
    # Create separate optimizers for each batch
    optimizers = []
    for i in range(n_restarts):
        # Only use Adam with different learning rates
        lr_factor = 1.0 + 0.2 * (i % 3)  # Still vary learning rates
        opt = torch.optim.Adam([expanded_states_list[i]], lr=learning_rate * lr_factor)
        optimizers.append(opt)
    
    print(f"Optimizing from {n_inits * n_restarts} initial points in {n_restarts} batches")
    
    # Run the optimization with simulated annealing
    iter_count = 1
    q_prev_list = [torch.full((states_batch.shape[0],), float("nan"), device=device) 
                   for states_batch in expanded_states_list]
    patience = 0
    max_patience = 10  # For early stopping if no improvement
    
    # Track best states found so far
    best_states_list = [states_batch.clone().detach() for states_batch in expanded_states_list]
    best_q_list = [torch.full((states_batch.shape[0],), float("inf"), device=device) 
                   for states_batch in expanded_states_list]
    
    # Keep track of overall best states across all batches
    all_best_states = []
    all_best_q = []

    
    while iter_count <= max_iters:
        # Current temperature for simulated annealing
        current_temp = temperature * (1.0 - min(0.95, iter_count / max_iters))
        
        # Process each batch separately
        mean_q_all = 0
        mean_dq_all = 0
        dq_list = []
        q_list = []
        
        # Process each batch individually
        for i in range(n_restarts):
            # Compute q and dq for the current batch of states
            F = model(expanded_inputs_list[i], expanded_states_list[i])
            q = 0.5 * torch.sum((F.squeeze() - expanded_states_list[i].squeeze()) ** 2, dim=1)
            q_list.append(q)
            
            # Track best states in this batch
            improved_mask = q < best_q_list[i]
            if torch.any(improved_mask):
                best_states_list[i][improved_mask] = expanded_states_list[i][improved_mask].clone().detach()
                best_q_list[i][improved_mask] = q[improved_mask]
                patience = 0
            
            # Calculate change in q
            dq = torch.abs(q - q_prev_list[i]) if iter_count > 1 else torch.ones_like(q)
            dq_list.append(dq)
            
            # Optimize this batch with Adam
            optimizers[i].zero_grad()
            q_mean = torch.mean(q)
            mean_q_all += q_mean.item() / n_restarts
            mean_dq_all += torch.mean(dq).item() / n_restarts
            q_mean.backward()
            optimizers[i].step()
            
            # Update previous q values
            q_prev_list[i] = q.detach()
            
            # Periodically reinject noise (simulated annealing style)
            if iter_count % 100 == 0:
                with torch.no_grad():
                    noise = current_temp * torch.randn_like(expanded_states_list[i])
                    # Only add noise to states that aren't converging well
                    noise_mask = (q > q_threshold).unsqueeze(1)
                    expanded_states_list[i].data = expanded_states_list[i].data + noise * noise_mask
        
        # Report progress
        if iter_count % 500 == 0:
            print(f"\nIteration {iter_count}/{max_iters}")
            print(f"Mean q = {mean_q_all:.2E}")
            print(f"Mean dq = {mean_dq_all:.2E}")
            
            # Find the overall best q across all batches
            best_q_value = float('inf')
            for bq in best_q_list:
                batch_min = torch.min(bq).item()
                if batch_min < best_q_value:
                    best_q_value = batch_min
            
            print(f"Best q = {best_q_value:.2E}")
            print(f"Temperature = {current_temp:.2E}")
            
            # Occasionally swap states between batches
            if iter_count % 1000 == 0 and n_restarts > 1:
                with torch.no_grad():
                    # Find the batch with the best and worst states
                    best_batch_idx = 0
                    worst_batch_idx = 0
                    best_q_val = float('inf')
                    worst_q_val = float('-inf')
                    
                    for b_idx, bq in enumerate(best_q_list):
                        batch_mean = torch.mean(bq).item()
                        if batch_mean < best_q_val:
                            best_q_val = batch_mean
                            best_batch_idx = b_idx
                        if batch_mean > worst_q_val:
                            worst_q_val = batch_mean
                            worst_batch_idx = b_idx
                    
                    # If there's a clear difference, move some good states to replace bad states
                    if best_batch_idx != worst_batch_idx:
                        # Sort states in best batch by q value
                        best_q_batch = q_list[best_batch_idx]
                        _, indices = torch.sort(best_q_batch)
                        
                        # Get worst states in worst batch
                        worst_q_batch = q_list[worst_batch_idx]
                        _, worst_indices = torch.sort(worst_q_batch, descending=True)
                        
                        # Replace some worst states with good states + noise
                        num_to_replace = min(n_inits // 4, len(indices), len(worst_indices))
                        for j in range(num_to_replace):
                            good_state = expanded_states_list[best_batch_idx][indices[j]].clone()
                            expanded_states_list[worst_batch_idx][worst_indices[j]] = good_state + 0.05 * torch.randn_like(good_state)
        
        # Early stopping criteria
        early_stop = (mean_dq_all < early_stop_threshold and iter_count > 1000) or patience >= max_patience
        if early_stop:
            print(f"Converged at iteration {iter_count}. Early stopping.")
            break
        
        iter_count += 1
    
    # Collect all the best states from all batches
    all_best_states = torch.cat(best_states_list, dim=0)
    all_best_q = torch.cat(best_q_list, dim=0)
    
    # Collect corresponding inputs
    all_best_inputs = torch.cat(expanded_inputs_list, dim=0)
    
    # Evaluate final fixed points
    with torch.no_grad():
        F_final = model(all_best_inputs, all_best_states)
        q_final = 0.5 * torch.sum((F_final.squeeze() - all_best_states.squeeze()) ** 2, dim=1)
    
    # Filter to keep only actual fixed points based on threshold
    fixed_point_mask = q_final < q_threshold
    filtered_states = all_best_states[fixed_point_mask]
    filtered_q = q_final[fixed_point_mask]
    
    # Create a placeholder for dq
    filtered_dq = torch.zeros_like(filtered_q)
    for i, dq_batch in enumerate(dq_list):
        batch_size = dq_batch.shape[0]
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        if start_idx < all_best_q.shape[0]:
            end_idx = min(end_idx, all_best_q.shape[0])
            batch_mask = fixed_point_mask[start_idx:end_idx]
            if torch.any(batch_mask):
                filtered_dq_part = dq_batch[batch_mask]
                # Only copy if dimensions match
                if len(filtered_dq_part) <= len(filtered_dq):
                    filtered_dq[:len(filtered_dq_part)] = filtered_dq_part
    
    # If no points meet threshold, take the best n_inits points
    if filtered_states.shape[0] < 10:
        print(f"Only {filtered_states.shape[0]} points below threshold. Taking top {min(n_inits, all_best_states.shape[0])} best points.")
        topk_values, topk_indices = torch.topk(q_final, min(n_inits, all_best_states.shape[0]), largest=False)
        filtered_states = all_best_states[topk_indices]
        filtered_q = q_final[topk_indices]
        filtered_dq = torch.zeros_like(filtered_q)  # Reset dq since we can't easily match indices
    
    # Create the FixedPoints object with the filtered results
    filtered_states_np = filtered_states.cpu().detach().numpy().squeeze()
    filtered_q_np = filtered_q.cpu().detach().numpy()
    filtered_dq_np = filtered_dq.cpu().detach().numpy()
    
    # Make sure we have matching sizes for initial states
    initial_states_subset = initial_states
    if len(filtered_states_np) < len(initial_states):
        # Truncate initial states to match filtered states
        initial_states_subset = initial_states[:len(filtered_states_np)]
    elif len(filtered_states_np) > len(initial_states):
        # Pad initial states if needed (shouldn't happen normally)
        padding = np.zeros((len(filtered_states_np) - len(initial_states),) + initial_states.shape[1:])
        initial_states_subset = np.concatenate([initial_states, padding], axis=0)
    
    all_fps = FixedPoints(
        xstar=filtered_states_np,
        x_init=initial_states_subset,
        qstar=filtered_q_np,
        dq=filtered_dq_np,
        n_iters=np.full_like(filtered_q_np, iter_count),
    )

    print(f"Found {len(all_fps.xstar)} unique fixed points.")
    
    if compute_jacobians:
        # Compute the Jacobian for each fixed point
        def J_func(model, inputs_, x):
            # This function takes both the additional inputs and the state.
            F = model(inputs_, x)
            return F.squeeze()

        def compute_jacobians_func(model, inputs, x_data):
            all_J = []
            x = torch.tensor(x_data, device=device)
            
            # Create appropriate inputs for the fixed points
            if len(inputs.shape) == 2:  # Regular 2D input tensor
                if inputs.shape[0] == 1:
                    inputs = inputs.repeat(x.shape[0], 1)
                elif inputs.shape[0] != x.shape[0]:
                    # Take the first n inputs or repeat as needed
                    if inputs.shape[0] < x.shape[0]:
                        repeats = (x.shape[0] + inputs.shape[0] - 1) // inputs.shape[0]
                        inputs = inputs.repeat(repeats, 1)
                    inputs = inputs[:x.shape[0]]

            for i in range(x.size(0)):
                if len(inputs.shape) == 2:
                    inputs_1 = inputs[i, :].unsqueeze(0)
                else:
                    inputs_1 = inputs.unsqueeze(0) if inputs.dim() == 1 else inputs
                
                single_x = x[i, :].unsqueeze(0)

                J = torch.autograd.functional.jacobian(
                    lambda x: J_func(model, inputs_1, x), single_x
                )
                all_J.append(J.squeeze())

            return all_J

        # For jacobian computation, just use the original input
        # This ensures we're evaluating the stability at the right input conditions
        all_J = compute_jacobians_func(model, inputs[:1], all_fps.xstar)
        
        # Recombine and decompose Jacobians for the whole batch
        if all_J:
            dFdx = torch.stack(all_J).cpu().detach().numpy()
            all_fps.J_xstar = dFdx
            all_fps.decompose_jacobians()

            return all_fps
        else:
            return []
    else:
        return all_fps

def find_fixed_points_old(
    model: pl.LightningModule,
    state_trajs: np.array,
    inputs: np.array,
    n_inits=1024,
    noise_scale=0.0,
    learning_rate=1e-2,
    max_iters=10000,
    device="cpu",
    seed=0,
    compute_jacobians=False,
):
    # set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    inputs = torch.tensor(inputs, device=device, dtype=torch.float32)
    state_trajs = torch.tensor(state_trajs, device=device,  dtype=torch.float32)

    model = model.to(device)
    state_trajs = state_trajs.to(device)
    inputs = inputs.to(device)

    # Prevent gradient computation for the neural ODE
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Choose random points along the observed trajectories
    if len(state_trajs.shape) > 2:
        n_samples, n_steps, state_dim = state_trajs.shape
        state_pts = state_trajs.reshape(-1, state_dim)
        if len(inputs.shape) > 1:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        idx = torch.randint(n_samples * n_steps, size=(n_inits,), device=device)
    else:
        n_samples_steps, state_dim = state_trajs.shape
        state_pts = state_trajs
        idx = torch.randint(n_samples_steps, size=(n_inits,), device=device)

    # Select the initial states
    states = state_pts[idx]
    if len(inputs.shape) > 1:
        inputs = inputs[idx]
    else:
        inputs = inputs.unsqueeze(0).repeat(n_inits, 1)

    # Add Gaussian noise to the sampled points
    states = states + noise_scale * torch.randn_like(states, device=device)

    # Require gradients for the states
    states = states.detach()
    initial_states = states.detach().cpu().numpy()
    states.requires_grad = True

    # Create the optimizer
    opt = torch.optim.Adam([states], lr=learning_rate)

    # Run the optimization
    iter_count = 1
    q_prev = torch.full((n_inits,), float("nan"), device=device)
    while True:
        # Compute q and dq for the current states
        F = model(inputs, states)
        q = 0.5 * torch.sum((F.squeeze() - states.squeeze()) ** 2, dim=1)
        dq = torch.abs(q - q_prev)
        q_scalar = torch.mean(q)

        # Backpropagate gradients and optimize
        q_scalar.backward()
        opt.step()
        opt.zero_grad()

        # Detach evaluation tensors
        q_np = q.cpu().detach().numpy()
        dq_np = dq.cpu().detach().numpy()
        # Report progress
        if iter_count % 500 == 0:
            mean_q, std_q = np.mean(q_np), np.std(q_np)
            mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
            print(f"\nIteration {iter_count}/{max_iters}")
            print(f"q = {mean_q:.2E} +/- {std_q:.2E}")
            print(f"dq = {mean_dq:.2E} +/- {std_dq:.2E}")

        # Check termination criteria
        if iter_count + 1 > max_iters:
            print("Maximum iteration count reached. Terminating.")
            break
        q_prev = q
        iter_count += 1
    # Collect fixed points

    qstar = q.cpu().detach().numpy()
    all_fps = FixedPoints(
        xstar=states.cpu().detach().numpy().squeeze(),
        x_init=initial_states,
        qstar=qstar,
        dq=dq.cpu().detach().numpy(),
        n_iters=np.full_like(qstar, iter_count),
    )

    print(f"Found {len(all_fps.xstar)} unique fixed points.")
    if compute_jacobians:
        # Compute the Jacobian for each fixed point
        def J_func(model, inputs_, x):
            # This function takes both the additional inputs and the state.
            F = model(inputs_, x)
            return F.squeeze()

        def compute_jacobians_func(model, inputs, x_data):
            all_J = []
            x = torch.tensor(x_data, device=device)

            for i in range(x.size(0)):
                inputs_1 = inputs[i, :].unsqueeze(0)
                single_x = x[i, :].unsqueeze(0)

                J = torch.autograd.functional.jacobian(
                    lambda x: J_func(model, inputs_1, x), single_x
                )
                all_J.append(J.squeeze())

            return all_J

        all_J = compute_jacobians_func(model, inputs, all_fps.xstar)
        # Recombine and decompose Jacobians for the whole batch
        if all_J:
            dFdx = torch.stack(all_J).cpu().detach().numpy()
            all_fps.J_xstar = dFdx
            all_fps.decompose_jacobians()

            return all_fps
        else:
            return []
    else:
        return all_fps


def find_fixed_points_coupled(
    model: pl.LightningModule,
    context_inputs: np.array,
    env_states: np.array,
    model_states: np.array,
    joint_states: np.array,
    n_inits=1024,
    noise_scale=0.0,
    learning_rate=1e-2,
    max_iters=10000,
    device="cpu",
    seed=0,
    compute_jacobians=False,
):
    # set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    model_states = model_states.to(device)
    env_states = env_states.to(device)
    context_inputs = context_inputs.to(device)
    joint_states = joint_states.to(device)

    # Model takes in "model_input" and "hidden"
    # Model input is the concatenation of
    # the environment states and the context inputs (in that order)
    # Hidden is the hidden state of the model

    rand_inds = torch.randint(0, env_states.size(0), (n_inits,), device=device)
    env_states = env_states[rand_inds]
    model_states = model_states[rand_inds]
    context_inputs = context_inputs[rand_inds]
    joint_states = joint_states[rand_inds]

    env_states = env_states.detach() + noise_scale * torch.randn_like(
        env_states, device=device
    )
    model_states = model_states.detach() + noise_scale * torch.randn_like(
        model_states, device=device
    )

    env_states.requires_grad = True
    model_states.requires_grad = True
    # Create the optimizer
    opt = torch.optim.Adam([env_states, model_states], lr=learning_rate)
    initial_states = torch.cat((env_states, model_states), dim=1).detach().cpu().numpy()

    # Run the optimization
    iter_count = 1
    q_model_prev = torch.full((n_inits,), float("nan"), device=device)
    q_env_prev = torch.full((n_inits,), float("nan"), device=device)
    while True:
        # Compute q and dq for the current states
        (
            action,
            hidden_step,
            env_states_step,
            joint_states_step,
        ) = model.forward_step_coupled(
            env_states, context_inputs, model_states, joint_states
        )

        q_model = 0.5 * torch.sum(
            (hidden_step.squeeze() - model_states.squeeze()) ** 2, dim=1
        )
        q_env = 0.5 * torch.sum(
            (env_states_step.squeeze() - env_states.squeeze()) ** 2, dim=1
        )

        dq_model = torch.abs(q_model - q_model_prev)
        dq_env = torch.abs(q_env - q_env_prev)

        q_model_scalar = torch.mean(q_model)
        q_env_scalar = torch.mean(q_env)

        q_scalar = q_model_scalar + q_env_scalar
        q = q_model + q_env
        dq = dq_model + dq_env

        # Backpropagate gradients and optimize
        q_scalar.backward()
        opt.step()
        opt.zero_grad()

        # Detach evaluation tensors
        q_np = q.cpu().detach().numpy()
        dq_np = dq.cpu().detach().numpy()
        # Report progress
        if iter_count % 10 == 0:
            mean_q, std_q = np.mean(q_np), np.std(q_np)
            mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
            print(f"\nIteration {iter_count}/{max_iters}")
            print(f"q = {mean_q:.2E} +/- {std_q:.2E}")
            print(f"dq = {mean_dq:.2E} +/- {std_dq:.2E}")

        # Check termination criteria
        if iter_count + 1 > max_iters:
            print("Maximum iteration count reached. Terminating.")
            break
        q_model_prev = q_model
        q_env_prev = q_env
        iter_count += 1
    # Collect fixed points
    states = torch.cat((env_states, model_states), dim=1)
    qstar = q.cpu().detach().numpy()
    all_fps = FixedPoints(
        xstar=states.cpu().detach().numpy().squeeze(),
        x_init=initial_states,
        qstar=qstar,
        dq=dq.cpu().detach().numpy(),
        n_iters=np.full_like(qstar, iter_count),
    )

    print(f"Found {len(all_fps.xstar)} unique fixed points.")
    if compute_jacobians:  # TODO: Fix this
        # Compute the Jacobian for each fixed point
        def J_func(model, inputs_, x):
            # This function takes both the additional inputs and the state.
            F = model(inputs_, x)
            return F.squeeze()

        def compute_jacobians_func(model, inputs, x_data):
            all_J = []
            x = torch.tensor(x_data, device=device)

            for i in range(x.size(0)):
                inputs_1 = inputs[i, :].unsqueeze(0)
                single_x = x[i, :].unsqueeze(0)

                J = torch.autograd.functional.jacobian(
                    lambda x: J_func(model, inputs_1, x), single_x
                )
                all_J.append(J.squeeze())

            return all_J

        all_J = compute_jacobians_func(model, all_fps.xstar)
        # Recombine and decompose Jacobians for the whole batch
        if all_J:
            dFdx = torch.stack(all_J).cpu().detach().numpy()
            all_fps.J_xstar = dFdx
            all_fps.decompose_jacobians()

            return all_fps
        else:
            return []
    else:
        return all_fps


def find_fixed_points_dt(
    model: pl.LightningModule,
    state_trajs: np.array,
    inputs: np.array,
    n_inits=1024,
    noise_scale=0.0,
    learning_rate=1e-2,
    max_iters=10000,
    device="cpu",
    seed=0,
    compute_jacobians=False,
):
    # set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)
    state_trajs = state_trajs.to(device)
    inputs = inputs.to(device)

    # Prevent gradient computation for the neural ODE
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Choose random points along the observed trajectories
    if len(state_trajs.shape) > 2:
        n_samples, n_steps, state_dim = state_trajs.shape
        state_pts = state_trajs.reshape(-1, state_dim)
        if len(inputs.shape) > 1:
            inputs = inputs.reshape(-1, inputs.shape[-1])
        idx = torch.randint(n_samples * n_steps, size=(n_inits,), device=device)
    else:
        n_samples_steps, state_dim = state_trajs.shape
        state_pts = state_trajs
        idx = torch.randint(n_samples_steps, size=(n_inits,), device=device)

    # Select the initial states
    states = state_pts[idx]
    if len(inputs.shape) > 1:
        inputs = inputs[idx]
    else:
        inputs = inputs.unsqueeze(0).repeat(n_inits, 1)

    # Add Gaussian noise to the sampled points
    states = states + noise_scale * torch.randn_like(states, device=device)

    # Require gradients for the states
    states = states.detach()
    initial_states = states.detach().cpu().numpy()
    states.requires_grad = True

    # Create the optimizer
    opt = torch.optim.Adam([states], lr=learning_rate)

    # Run the optimization
    iter_count = 1
    q_prev = torch.full((n_inits,), float("nan"), device=device)
    x_store = np.zeros((n_inits, max_iters, state_dim))
    q_store = np.zeros((n_inits, max_iters))
    while True:
        # Compute q and dq for the current states
        x_store[:, iter_count - 1, :] = states.cpu().detach().numpy()
        q_store[:, iter_count - 1] = q_prev.cpu().detach().numpy()
        _, F = model.decoder(inputs, states)
        q = 0.5 * torch.sum((F.squeeze() - states.squeeze()) ** 2, dim=1)
        dq = torch.abs(q - q_prev)
        q_scalar = torch.mean(q)

        # Backpropagate gradients and optimize
        q_scalar.backward()
        opt.step()
        opt.zero_grad()

        # Detach evaluation tensors
        q_np = q.cpu().detach().numpy()
        dq_np = dq.cpu().detach().numpy()
        # Report progress
        if iter_count % 500 == 0:
            mean_q, std_q = np.mean(q_np), np.std(q_np)
            mean_dq, std_dq = np.mean(dq_np), np.std(dq_np)
            print(f"\nIteration {iter_count}/{max_iters}")
            print(f"q = {mean_q:.2E} +/- {std_q:.2E}")
            print(f"dq = {mean_dq:.2E} +/- {std_dq:.2E}")

        # Check termination criteria
        if iter_count + 1 > max_iters:
            print("Maximum iteration count reached. Terminating.")
            break
        q_prev = q
        q_store[:, iter_count - 1] = q_prev.cpu().detach().numpy()
        iter_count += 1
    # Collect fixed points

    qstar = q.cpu().detach().numpy()
    all_fps = FixedPoints(
        xstar=states.cpu().detach().numpy().squeeze(),
        x_init=initial_states,
        qstar=qstar,
        dq=dq.cpu().detach().numpy(),
        n_iters=np.full_like(qstar, iter_count),
    )

    print(f"Found {len(all_fps.xstar)} unique fixed points.")
    if compute_jacobians:
        # Compute the Jacobian for each fixed point
        def J_func(model, inputs_, x):
            # This function takes both the additional inputs and the state.
            _, F = model(inputs_, x)
            return F.squeeze()

        def compute_jacobians_func(model, inputs, x_data):
            all_J = []
            x = torch.tensor(x_data, device=device)

            for i in range(x.size(0)):
                inputs_1 = inputs[i, :].unsqueeze(0)
                single_x = x[i, :].unsqueeze(0)

                J = torch.autograd.functional.jacobian(
                    lambda x: J_func(model, inputs_1, x), single_x
                )
                all_J.append(J.squeeze())

            return all_J

        all_J = compute_jacobians_func(model, inputs, all_fps.xstar)
        # Recombine and decompose Jacobians for the whole batch
        if all_J:
            dFdx = torch.stack(all_J).cpu().detach().numpy()
            all_fps.J_xstar = dFdx
            all_fps.decompose_jacobians()

            return all_fps
        else:
            return []
    else:
        return all_fps
