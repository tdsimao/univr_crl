import gym



if __name__ == "__main__":
    env = gym.make('gym_factored:taxi-fuel-v0')
    ob = env.reset()
    decoded_state = list(env.decode(ob))
    assert ob == env.encode(*decoded_state)
    while True:
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        decoded_state = list(env.decode(ob))
        print(f"state {decoded_state}\taction {action}\treward={reward}\tcost={info['cost']}")
        assert ob == env.encode(*decoded_state)
        if done:
            break
