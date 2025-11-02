import streamlit as st
import pandas as pd
import numpy as np
import random

st.title("📺 TV Program Scheduling using Genetic Algorithm")

# --- Step 1: Upload CSV ---
uploaded_file = st.file_uploader("Upload your program ratings CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data Preview")
    st.write(data.head())

    programs = data.iloc[:, 0].tolist()
    ratings = data.iloc[:, 1:].values.tolist()

    # Dynamically extract hours from CSV header
    hours = list(data.columns[1:])
    num_hours = len(hours)

    # --- Step 2: Genetic Algorithm Parameters ---
    st.sidebar.header("⚙️ Genetic Algorithm Settings")
    population_size = st.sidebar.slider("Population Size", 10, 200, 50)
    generations = st.sidebar.slider("Generations", 10, 500, 100)
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)

    # --- Step 3: Fitness Function ---
    def fitness(schedule):
        total_rating = sum(ratings[schedule[h]][h] for h in range(num_hours))
        return total_rating

    # --- Step 4: Initialize Population ---
    def initialize_population():
        return [random.sample(range(len(programs)), num_hours) for _ in range(population_size)]

    # --- Step 5: Selection ---
    def selection(population):
        return sorted(population, key=lambda s: fitness(s), reverse=True)[:2]

    # --- Step 6: Crossover ---
    def crossover(parent1, parent2):
        point = random.randint(1, num_hours - 2)
        child = parent1[:point] + [p for p in parent2 if p not in parent1[:point]]
        return child

    # --- Step 7: Mutation ---
    def mutate(schedule):
        if random.random() < mutation_rate:
            i, j = random.sample(range(num_hours), 2)
            schedule[i], schedule[j] = schedule[j], schedule[i]
        return schedule

    # --- Step 8: Run Algorithm ---
    if st.button("Run Genetic Algorithm"):
        population = initialize_population()

        for _ in range(generations):
            new_population = selection(population)
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(selection(population), 2)
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            population = new_population

        best_schedule = selection(population)[0]
        best_fitness = fitness(best_schedule)

        # --- Step 9: Display Results ---
        result_df = pd.DataFrame({
            "Hour": hours,
            "Program": [programs[i] for i in best_schedule],
            "Rating": [ratings[i][h] for h, i in enumerate(best_schedule)]
        })

        st.subheader("🏆 Optimal TV Schedule")
        st.write(result_df)
        st.success(f"✅ Total Rating: {best_fitness:.2f}")
