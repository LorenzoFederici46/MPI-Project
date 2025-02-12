import matplotlib.pyplot as plt

# Numero di processi MPI utilizzati
processi = [1, 2, 4, 8]

# Tempi di esecuzione misurati per ogni numero di processi
tempi = [10.5, 6.0, 3.8, 2.5]  # Sostituisci con i tuoi dati reali

# Calcolo dello speedup: Speedup = Tempo seriale / Tempo parallelo
speedup = [tempi[0] / t for t in tempi]

# Creazione del grafico
plt.plot(processi, speedup, marker='o', linestyle='-', color='b', label="Speedup MPI")

# Etichette e titolo
plt.xlabel("Numero di processi")
plt.ylabel("Speedup")
plt.title("Speedup MPI per kNN")
plt.legend()
plt.grid()

# Mostra il grafico
plt.show()
