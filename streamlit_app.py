import streamlit as st
import pandas as pd
import random

# Title
st.title("📺 Genetic Algorithm - TV Program Scheduling Optimizer")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("program_ratings_modified.csv")
    return df

df = load_data()
st.subheader("📊 Program Ratings Dataset")
st.dataframe(df)

# Parameters input
st.sidebar.header("🧬 Genetic Algorithm Parameters")
CO_R = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.8)
MUT_R = st.sidebar.slider("Mutation Rate", 0.01, 0.1, 0.02)
GEN = st.sidebar.slider("Number of Generations", 10, 200, 60)
POP = st.sidebar.slider("Population Size", 10, 200, 60)

st.sidebar.write("---")

# Prepare data
programs = df["Type of Program"].tolist()
ratings = df.drop(columns=["Type of Program"]).values
hours = df.columns[1:]
num_hours = len(hours)

# Fitness function
def fitness(schedule):
    total = 0
    for h in range(num_hours):
        total += ratings[schedule[h]][h]
    return total

# Generate population
def generate_population():
    population = []
    for _ in range(POP):
        # each schedule length must equal number of hours (not programs)
        population.append(random.sample(range(len(programs)), num_hours))
    return population

# Selection (tournament)
def select(population):
    a, b = random.sample(population, 2)
    return a if fitness(a) > fitness(b) else b

# Crossover (single point)
def crossover(parent1, parent2):
    point = random.randint(1, num_hours - 2)
    child = parent1[:point] + [p for p in parent2 if p not in parent1[:point]]
    # if child shorter than num_hours, fill with random remaining
    while len(child) < num_hours:
        remaining = [p for p in range(len(programs)) if p not in child]
        child.append(random.choice(remaining))
    return child

# Mutation (swap)
def mutate(individual):
    if random.random() < MUT_R:
        i, j = random.sample(range(num_hours), 2)
        individual[i], individual[j] = individual[j], individual[i]
    return individual

# Run GA
def genetic_algorithm():
    population = generate_population()
    best = None
    best_fit = -1
    for g in range(GEN):
        new_pop = []
        for _ in range(POP):
            p1, p2 = select(population), select(population)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop
        current_best = max(population, key=fitness)
        current_fit = fitness(current_best)
        if current_fit > best_fit:
            best_fit = current_fit
            best = current_best
    return best, best_fit

if st.button("🚀 Run Genetic Algorithm"):
    best_schedule, total_rating = genetic_algorithm()

    result_df = pd.DataFrame({
        "Hour": hours,
        "Program": [programs[i] for i in best_schedule],
        "Rating": [ratings[i][h] for h, i in enumerate(best_schedule)]
    })

    st.success(f"✅ Best Total Rating: {total_rating:.3f}")
    st.subheader("📅 Optimized TV Schedule")
    st.dataframe(result_df)

    st.download_button(
        "💾 Download Schedule as CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        "optimized_schedule.csv",
        "text/csv"
    )

st.markdown("---")
st.caption("Developed by Rupha | Genetic Algorithm TV Scheduling Project")
