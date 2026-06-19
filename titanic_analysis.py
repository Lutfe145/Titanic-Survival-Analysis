import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("titanic.csv")

# =========================
# BASIC INFORMATION
# =========================
print("\n===== TITANIC DATASET INFO =====\n")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# SURVIVAL RATE
# =========================
survival_rate = df["Survived"].mean() * 100

print("\nOverall Survival Rate:")
print(f"{survival_rate:.2f}%")

# =========================
# SURVIVAL DISTRIBUTION
# =========================
plt.figure(figsize=(6, 4))
df["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# =========================
# GENDER VS SURVIVAL
# =========================
gender_survival = pd.crosstab(df["Sex"], df["Survived"])

plt.figure(figsize=(6, 4))
gender_survival.plot(kind="bar")
plt.title("Gender vs Survival")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

print("\nGender vs Survival:")
print(gender_survival)

# =========================
# PASSENGER CLASS VS SURVIVAL
# =========================
class_survival = pd.crosstab(df["Pclass"], df["Survived"])

plt.figure(figsize=(6, 4))
class_survival.plot(kind="bar")
plt.title("Passenger Class vs Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

print("\nPassenger Class vs Survival:")
print(class_survival)

# =========================
# AGE DISTRIBUTION
# =========================
plt.figure(figsize=(8, 4))
df["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# =========================
# FARE DISTRIBUTION
# =========================
plt.figure(figsize=(8, 4))
df["Fare"].hist(bins=20)
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

# =========================
# SURVIVAL RATE BY CLASS
# =========================
survival_by_class = df.groupby("Pclass")["Survived"].mean() * 100

print("\nSurvival Rate by Passenger Class:")
print(survival_by_class)

plt.figure(figsize=(6, 4))
survival_by_class.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.show()

print("\n===== ANALYSIS COMPLETED SUCCESSFULLY =====")