import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# -------------------------------
# 1. Basic setup
# -------------------------------
data_root = "./data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs_teacher = 3       # quick demo
epochs_student = 5

# -------------------------------
# 2. Dataset & DataLoader
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = datasets.MNIST(root=data_root, train=True, download=False, transform=transform)
testset  = datasets.MNIST(root=data_root, train=False, download=False, transform=transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(testset, batch_size=1000, shuffle=False)

# -------------------------------
# 3. Teacher & Student networks
# -------------------------------
class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(4608, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

teacher = TeacherNet().to(device)
student = StudentNet().to(device)

# -------------------------------
# 4. Training utilities
# -------------------------------
def train(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

def test(model, loader, name="Model"):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += pred.eq(y).sum().item()
    acc = 100. * correct / len(loader.dataset)
    print(f"{name} accuracy: {acc:.2f}%")
    return acc

# -------------------------------
# 5. Train teacher first
# -------------------------------
opt_teacher = optim.Adam(teacher.parameters(), lr=1e-3)
print("Training teacher...")
for epoch in range(1, epochs_teacher+1):
    train(teacher, train_loader, opt_teacher, epoch)
test(teacher, test_loader, name="Teacher")

# Save teacher for reuse
os.makedirs("log/10-16/checkpoints", exist_ok=True)
torch.save(teacher.state_dict(), "log/10-16/checkpoints/teacher_mnist.pt")

# -------------------------------
# 6. Define distillation loss
# -------------------------------
def distillation_loss(student_logits, teacher_logits, labels, T=4.0, alpha=0.5):
    hard_loss = F.cross_entropy(student_logits, labels)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)
    return alpha * hard_loss + (1 - alpha) * soft_loss

# -------------------------------
# 7. Train student with distillation
# -------------------------------
teacher.load_state_dict(torch.load("log/10-16/checkpoints/teacher_mnist.pt"))
teacher.eval()
opt_student = optim.Adam(student.parameters(), lr=1e-3)

print("\nTraining student with distillation...")
for epoch in range(1, epochs_student+1):
    student.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            teacher_logits = teacher(x)
        student_logits = student(x)
        loss = distillation_loss(student_logits, teacher_logits, y, T=4.0, alpha=0.7)
        opt_student.zero_grad()
        loss.backward()
        opt_student.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: distill_loss={total_loss/len(train_loader):.4f}")

# -------------------------------
# 8. Evaluate
# -------------------------------
test(student, test_loader, name="Student")
