from DQLN import Agent 
import numpy as np
import matplotlib.pyplot as plt
from CardGameEnv import CardGameEnv  # Import your custom environment
import traceback
import os

def plotLearning(x, scores, eps_history, filename):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot scores
    ax1.plot(x, scores, 'b-', alpha=0.7, label='Score')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.set_title('Card Game Training Scores')
    ax1.grid(True)
    ax1.legend()
    
    # Plot epsilon values
    ax2.plot(x, eps_history, 'r-', alpha=0.7, label='Epsilon')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Epsilon Decay')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_agent(env, agent, n_eval_episodes=10):
    """Evaluate the trained agent"""
    total_rewards = []
    wins = 0
    
    # Temporarily set epsilon to 0 for evaluation (no exploration)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(n_eval_episodes):
        observation, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.choose_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        
        total_rewards.append(total_reward)
        if info.get('winner') == 'player':
            wins += 1
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    avg_reward = np.mean(total_rewards)
    win_rate = wins / n_eval_episodes
    
    return avg_reward, win_rate, total_rewards

if __name__ == '__main__':
    print("Starting Card Game RL Training...")
    
    try:
        # Create your custom environment
        print("Creating custom card game environment...")
        env = CardGameEnv()
        print(f"Environment created successfully.")
        print(f"Action space: {env.action_space.n} actions")
        print(f"Observation space: {env.observation_space.shape}")
        
        # Set up training parameters
        n_games = 1000  # More episodes for card games
        
        # Create agent with parameters adapted for your environment
        observation_dim = env.observation_space.shape[0]  # Get from your environment
        n_actions = env.action_space.n  # Get from your environment
        
        print("Creating agent...")
        agent = Agent(
            alpha=0.001,      # Learning rate - you might need to tune this
            gamma=0.95,       # Discount factor - slightly lower for card games
            n_actions=n_actions,
            epsilon=1.0,      # Start with full exploration
            batch_size=32,    # Smaller batch size might work better
            input_dims=observation_dim,
            epsilon_dec=0.995,  # Slower epsilon decay for more exploration
            epsilon_end=0.05,   # Keep some exploration even at the end
            mem_size=100000,    # Smaller memory for faster training
            fname='card_game_dqn.h5'
        )
        print("Agent created successfully")

        # Training loop
        scores = []
        eps_history = []
        win_rates = []
        
        # Evaluation frequency
        eval_frequency = 50  # Evaluate every 50 episodes
        
        for i in range(n_games):
            done = False
            score = 0
            observation, _ = env.reset()
            
            step_count = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                agent.remember(observation, action, reward, observation_, done)
                observation = observation_
                score += reward
                agent.learn()
                
                step_count += 1
                if step_count > 200:  # Prevent very long games
                    break
        
            eps_history.append(agent.epsilon)
            scores.append(score)
            
            # Calculate statistics
            avg_score = np.mean(scores[max(0, i-100):i+1])
            
            # Periodic evaluation
            if i % eval_frequency == 0 and i > 0:
                eval_reward, win_rate, _ = evaluate_agent(env, agent, n_eval_episodes=20)
                win_rates.append((i, win_rate))
                print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, '
                      f'Epsilon: {agent.epsilon:.3f}, Win Rate: {win_rate:.2f}, '
                      f'Winner: {info.get("winner", "unknown")}')
            else:
                print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}, '
                      f'Epsilon: {agent.epsilon:.3f}, Winner: {info.get("winner", "unknown")}')
            
            # Save model periodically
            if i % 100 == 0 and i > 0:
                try:
                    agent.save_model()
                    print(f"Model saved at episode {i}")
                except Exception as e:
                    print(f"Error saving model: {e}")

        # Final evaluation
        print("\nFinal Evaluation...")
        final_reward, final_win_rate, eval_scores = evaluate_agent(env, agent, n_eval_episodes=50)
        print(f"Final Average Reward: {final_reward:.3f}")
        print(f"Final Win Rate: {final_win_rate:.2f}")
        
        # Save final model
        try:
            print("Saving final model...")
            agent.save_model()
            print("Final model saved successfully")
        except Exception as e:
            print(f"Error saving final model: {e}")

        # Create plots
        print("Creating training plots...")
        filename = 'card_game_training.png'
        x = [i+1 for i in range(len(scores))]
        plotLearning(x, scores, eps_history, filename)
        
        # Plot win rates over time
        if win_rates:
            episodes, rates = zip(*win_rates)
            plt.figure(figsize=(10, 6))
            plt.plot(episodes, rates, 'g-', linewidth=2, label='Win Rate')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.title('Win Rate Over Time')
            plt.grid(True)
            plt.legend()
            plt.savefig('card_game_win_rates.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("Training completed successfully!")
        
        # Test the trained model
        print("\nTesting trained model...")
        test_episodes = 5
        for test_ep in range(test_episodes):
            observation, _ = env.reset()
            total_reward = 0
            done = False
            step = 0
            
            print(f"\n--- Test Episode {test_ep + 1} ---")
            env.render()
            
            while not done and step < 50:
                action = agent.choose_action(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
                step += 1
                
                print(f"Action: {action}, Reward: {reward:.3f}")
                env.render()
            
            print(f"Test Episode {test_ep + 1} - Total Reward: {total_reward:.3f}, Winner: {info.get('winner')}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()

print("Script finished")