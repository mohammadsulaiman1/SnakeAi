import torch 
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os
import pickle

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = .001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # Exploration-exploitation tradeoff
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)  # Experience replay buffer
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

        # Load the saved model if it exists
        model_path = "./model/model.pth"
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()  # Set model to evaluation mode
            print("Model loaded successfully.")
            
        if os.path.exists("memory.pkl"):
            try:
                with open("memory.pkl", "rb") as f:
                    self.memory = pickle.load(f)
                    if len(self.memory) > 0:
                        self.n_games = len(self.memory)  # Set n_games based on memory size
                        print(f"✅ Memory loaded successfully. {len(self.memory)} past experiences loaded.")
                        print(f"✅ Resuming from game {self.n_games}.")
            except Exception as e:
                print(f"❌ Error loading memory: {e}")

    
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached
        
        
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)    
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = max(5, 80 - (self.n_games * 0.75))  # Reduce randomness faster
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move
            
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        
        
        final_move = agent.get_action(state_old)
        
        reward, done, score = game.play_step(final_move)
        
        state_new = agent.get_state(game)
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)
        
        if done:
            # train the long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            
            # Save model every 10 games OR when a new record is achieved
            if score > record:
                record = score
                agent.model.save()
                print(f"New record! Model saved. Score: {score}")
            elif agent.n_games % 10 == 0:  # Every 10 games
                agent.model.save()
                print(f"Model progress saved at game {agent.n_games}.")
                
            if agent.n_games % 10 == 0:
                agent.model.save()

                # Save memory
                with open("memory.pkl", "wb") as f:
                    pickle.dump(agent.memory, f)

                print(f"Model and memory progress saved at game {agent.n_games}.")
                
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            
if __name__ == '__main__':
    train()