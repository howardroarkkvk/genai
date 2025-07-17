import torch
import torch.nn as nn

torch.manual_seed(42)
dropout=nn.Dropout(p=0.5)
print(dropout.training)
print(dropout.state_dict())

dropout.train()
batch_size, seq_length, embed_dim = 2, 5, 10
input = torch.randn(batch_size, seq_length, embed_dim)
print(input, input.shape)
output=dropout(input)
print(output, output.shape)

# dropout.eval()
# output1=dropout(input)
# print(output1,output1.shape)

# drop out helps in making network not to mug up the data so that during training it learns instead of mugging up
# it prevent overfitting...
# during training , each time some parts of the neurons gets switched off so that the other neurons learns and in the next iteration the neurons which swtiched off with be swithced on and other go 
# in swithced off mode, this way we will have model trained with multiple combinations of neurons  ....each weight is scaled to reflect the average contribution of all subnetworks..
# averaging the output of all mini networks --- like ensemeble learning....

# Great question! Let's break down how dropout acts like training many small networks and averaging them — a concept that might seem a bit magical at first but is quite intuitive once you get it.

# 🧠 What dropout does technically
# During training with dropout (say, p=0.5):

# In each forward pass, different random neurons are "dropped" (set to 0).

# This means the network architecture changes slightly every time — different combinations of neurons are used.

# Each variation is like a smaller sub-network inside the full network.

# 🧪 Analogy: Teamwork & Practice
# Imagine you have a large team of people working on solving a problem.

# Each time they practice, some team members are randomly told to sit out.

# So, the remaining ones must figure it out themselves, which forces the whole team to be better prepared.

# Over time, every possible combination of team members gets to work on the task.

# When it's time for the real job (inference), everyone works together, and you get a solution that reflects the collective knowledge of all those practice sessions.

# 🤖 In neural networks
# Dropout randomly disables neurons during training.

# This results in many different "thinned" subnetworks being trained.

# Each subnetwork learns slightly different features.

# At inference time, the full network is used, but each weight is scaled to reflect the average contribution of all subnetworks.

# This acts like averaging the outputs of all those mini-networks — a form of implicit ensemble learning.

# 📈 Why this is good:
# Ensembles are known to improve generalization — they smooth out individual model mistakes.

# Dropout gives you the benefit of an ensemble, without the cost of training and storing many separate models.

# 🔍 Visual intuition
# If you were to visualize the network:

# Training Pass	Active Neurons
# 1	[✓ ✓ ✗ ✓ ✗ ✓ ✗]
# 2	[✗ ✓ ✓ ✗ ✓ ✗ ✓]
# 3	[✓ ✗ ✓ ✓ ✗ ✓ ✓]

# Each row is a different subnetwork trained on the same data.
# Final inference is like averaging all their votes by using the full network with scaled weights.

# Let me know if you'd like to see this demonstrated in code or graphs!