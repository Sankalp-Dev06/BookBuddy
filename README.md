# BookBuddy Recommendation System

A book recommendation system built with deep reinforcement learning and PyTorch.

## Overview

BookBuddy uses a Deep Q-Network (DQN) to learn optimal book recommendation strategies. The system learns which books to recommend based on previously selected books, aiming to maximize factors such as relevance, diversity, and novelty.

## Features

- **DQN-powered recommendations**: Uses reinforcement learning to generate personalized book recommendations
- **Command-line interface**: Generate recommendations directly from the terminal
- **Web interface**: User-friendly web app to search books and get recommendations
- **GPU acceleration**: Optimized for fast training and inference with CUDA support

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Flask (for web interface)
- Pandas
- NumPy

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install torch pandas numpy flask
   ```

## Usage

### Training the Model

To train the recommendation model:

```bash
python train.py
```

This will train the DQN model and save checkpoints to `models/checkpoints/`.

### Command-line Recommendations

To generate recommendations from the command line:

```bash
python recommend.py --model models/checkpoints/dqn_recommender_best.pt --recommendations 5 --seed_books 1234 5678
```

Parameters:

- `--model`: Path to the trained model checkpoint
- `--recommendations`: Number of recommendations to generate (default: 5)
- `--seed_books`: List of book IDs to use as seed for recommendations
- `--user_id`: User ID to personalize recommendations (optional)
- `--verbose`: Show detailed information about recommended books

### Web Interface

To start the web interface:

```bash
python app.py
```

Then open your browser to http://localhost:5000

## Model Architecture

The recommendation system uses:

- **Deep Q-Network (DQN)**: A neural network that learns to predict the expected reward for each possible book recommendation
- **Experience Replay**: Stores agent experiences to improve learning efficiency
- **Target Network**: Stabilizes training by using a separate network for target value calculation

## Memory Optimization

The implementation includes several memory optimizations:

- Mixed precision training
- GPU memory management with periodic cache cleaning
- Safe model saving that avoids CUDA OOM errors
- CPU fallback for inference when GPU memory is constrained

## Contributors
-Ujjwal Chaurasia
-Vivek Aggarwal
-Sankalp Tiwari

## Preview
![image](https://github.com/user-attachments/assets/05cf4778-6bfa-4610-8c73-9920ea811600) ![image](https://github.com/user-attachments/assets/b5813a93-81cf-4ba9-8964-c81796285ac7)
![image](https://github.com/user-attachments/assets/28ed8773-ef86-4917-a33d-9a52381813b7) ![image](https://github.com/user-attachments/assets/9010e581-5b97-4baa-b398-23e629dab333)
![image](https://github.com/user-attachments/assets/9709bfa3-d3e5-4257-86e6-228d417a732a) ![image](https://github.com/user-attachments/assets/0ae4a25a-15eb-4f3d-b680-cf773f2f1d2b)
