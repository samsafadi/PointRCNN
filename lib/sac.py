def get_action(o, deterministic=False):
    return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                  deterministic)


# Set up function for computing SAC Q-losses
def compute_loss_q(data):
    o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

    q1 = ac.q1(o,a)
    q2 = ac.q2(o,a)

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        a2, logp_a2 = ac.pi(o2)

        # Target Q-values
        q1_pi_targ = ac_targ.q1(o2, a2)
        q2_pi_targ = ac_targ.q2(o2, a2)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    return loss_q, alpha * logp_a2

# Set up function for computing SAC pi loss
def compute_loss_pi(data):
    o = data['obs']
    pi, logp_pi = ac.pi(o)
    q1_pi = ac.q1(o, pi)
    q2_pi = ac.q2(o, pi)
    q_pi = torch.min(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = (alpha * logp_pi - q_pi).mean()

    # # Useful info for logging
    # pi_info = dict(LogPi=logp_pi.detach().numpy())

    return loss_pi, alpha * logp_pi


def update(batch_size, polyak=0.995,clip_grad=2.0):
    data = {}
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    data['obs']     = state.to(device)
    data['obs2']     = next_state.to(device)
    data['act']     = action.to(device)
    data['rew']     = reward.to(device)
    data['done']    = done.to(device)

    # First run one gradient descent step for Q1 and Q2
    q_optimizer.zero_grad()
    loss_q, _ = compute_loss_q(data)
    loss_q.backward()


    if clip_grad:
        if isinstance(clip_grad, bool):
            torch.nn.utils.clip_grad_norm_(q_params, max_norm=1.)
        else:
            torch.nn.utils.clip_grad_norm_(q_params, max_norm=clip_grad)
    q_optimizer.step()

    # Freeze Q-networks so you don't waste computational effort 
    # computing gradients for them during the policy learning step.
    for p in q_params:
        p.requires_grad = False

    # Next run one gradient descent step for pi.
    pi_optimizer.zero_grad()
    loss_pi, entropy = compute_loss_pi(data)
    loss_pi.backward()


    if clip_grad:
        if isinstance(clip_grad, bool):
            torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), max_norm=1.)
        else:
            torch.nn.utils.clip_grad_norm_(ac.pi.parameters(), max_norm=clip_grad)
    pi_optimizer.step()

    # Unfreeze Q-networks so you can optimize it at next DDPG step.
    for p in q_params:
        p.requires_grad = True

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)


def test_agent(num_test_episodes):
    eps_ret = 0 
    o = env.reset()
    for j in range(num_test_episodes):
        # o, d, ep_ret, ep_len = env.reset(), False, 0, 0
        d, ep_ret, ep_len =  False, 0, 0

        while not(d or (ep_len == max_ep_len)):
            # Take deterministic actions at test time 
            o, r, d, _ = env.step(get_action(o.view(1,-1), True))
            ep_ret += r
            ep_len += 1
        eps_ret += ep_ret
    return eps_ret/num_test_episodes




