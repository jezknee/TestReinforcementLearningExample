from DQLN import Agent 
import numpy as np
import gymnasium as gym 
import matplotlib.pyplot as plt
import traceback
import sys

def plotLearning(x, scores, eps_history, filename):
    print("Creating plot...")
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot scores
        ax1.plot(x, scores, 'b-', alpha=0.7, label='Score')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Score')
        ax1.set_title('Training Scores')
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
        print(f"Plot saved as {filename}")
        plt.show()
    except Exception as e:
        print(f"Error in plotting: {e}")
        traceback.print_exc()

print("Script starting...")

if __name__ == '__main__':
    print("Main block entered")
    
    try:
        print("Creating environment...")
        env = gym.make('LunarLander-v3')
        print(f"Environment created successfully. Action space: {env.action_space}, Observation space: {env.observation_space}")
        
       
        print("Creating agent...")
        agent = Agent(alpha=0.0005, gamma=0.99, n_actions=4, epsilon=1.0, batch_size=64, input_dims=8, epsilon_dec=0.996, epsilon_end=0.01, mem_size=1000000, fname='dqn_model.keras')
        print("Agent created successfully")

        scores = []
        eps_history = []

        for i in range(agent.batch_size):
            print(f"Starting episode {i}")
            done = False
            score = 0
            
            try:
                observation, info = env.reset()
                print(f"Episode {i} - Initial observation shape: {observation.shape}")
            except Exception as e:
                print(f"Error during env.reset(): {e}")
                continue
            
            step_count = 0
            while not done:
                try:
                    action = agent.choose_action(observation)
                    observation_, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    
                    agent.remember(observation, action, reward, observation_, done)
                    observation = observation_
                    score += reward
                    agent.learn()
                    
                    step_count += 1
                    if step_count > 1000:
                        print("Episode exceeded 1000 steps, ending...")
                        break
                        
                except Exception as e:
                    print(f"Error during step {step_count} of episode {i}: {e}")
                    traceback.print_exc()
                    break
        
            eps_history.append(agent.epsilon)
            scores.append(score)

            avg_score = np.mean(scores[max(0, i-100):i+1])
            print(f'episode {i}, score {score:.2f}, average score {avg_score:.2f}, epsilon {agent.epsilon:.3f}')
            
            if i % 10 == 0 and i > 0:
                try:
                    agent.save_model()
                    print(f"Model saved at episode {i}")
                except Exception as e:
                    print(f"Error saving model: {e}")

        print("Training completed, creating plot...")
        filename = 'lunarlander.png'
        x = [i+1 for i in range(len(scores))]
        plotLearning(x, scores, eps_history, filename)
        print("Script completed successfully")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

print("Script finished")