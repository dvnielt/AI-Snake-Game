# Snake AI with Deep Q-Learning

A Python implementation of the classic Snake game powered by a deep Q-learning neural network that learns to play autonomously.

## Features
- Self-learning AI using PyTorch
- Real-time visualization of gameplay
- Performance tracking with live score plotting
- Customizable neural network architecture

## Requirements
```
pygame
torch
numpy
matplotlib
IPython
```

## Installation
```bash
git clone [repository-url]
cd snake-ai
pip install -r requirements.txt
```

## Usage
Run the main script:
```bash
python agent.py
```

The program will:
1. Launch a pygame window showing the snake game
2. Display a plot tracking scores and learning progress
3. Save the best-performing model to `/model/model.pth`

## How It Works
The AI agent:
- Uses 11 input states (danger detection, movement direction, food location)
- Makes decisions through a neural network with 256 hidden nodes
- Learns via Q-learning with rewards for food collection (-10 for death, +10 for food)
- Balances exploration/exploitation using epsilon-greedy strategy

## Project Structure
- `agent.py`: Main training loop and AI logic
- `game.py`: Snake game implementation
- `model.py`: Neural network architecture
- `helper.py`: Visualization utilities

## Parameters
- `MAX_MEMORY`: 100,000 states
- `BATCH_SIZE`: 1,000 states for training
- `LR`: 0.001 learning rate
- `EPSILON`: Decays as games progress for exploration
- `GAMMA`: 0.9 discount rate

## Contributing
Feel free to fork, open issues, or submit PRs.
