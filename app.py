import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

from matplotlib.lines import Line2D

# ── Cargar y limpiar datos ──────────────────────────────────────────────────
df = pd.read_csv("data/global_climate_energy_2020_2024.csv", parse_dates=["date"])
df["country"] = df["country"].str.strip()

numeric_cols = [
    "avg_temperature", "co2_emission", "energy_price",
    "energy_consumption", "renewable_share", "industrial_activity_index"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["date", "country"] + numeric_cols).sort_values("date")


# ── Regresión lineal simple ─────────────────────────────────────────────────
def regression(xs, ys):
    n   = len(xs)
    mx  = np.mean(xs)
    my  = np.mean(ys)
    slope     = np.sum((xs - mx) * (ys - my)) / np.sum((xs - mx) ** 2)
    intercept = my - slope * mx
    ss_tot = np.sum((ys - my) ** 2)
    ss_res = np.sum((ys - (slope * xs + intercept)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
    return slope, intercept, r2

# ── CONFIGURACIÓN VISUAL GLOBAL (MISMA QUE CHART3) ───────────────
FIG_SIZE = (12, 5)

def apply_chart3_style(fig, ax):
    fig.set_size_inches(FIG_SIZE)

    # Espaciado igual al chart3
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.25)

    # Grid consistente
    ax.grid(axis="y", color="#b1acac", linestyle="--", linewidth=0.6, alpha=0.6)

    # Estilo de ticks
    ax.tick_params(axis="x", rotation=22, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 1 — Temperatura promedio vs Tiempo
# ══════════════════════════════════════════════════════════════════════════════
def chart1(df):
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["avg_temperature"]
        .mean()
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots()
    apply_chart3_style(fig, ax)   
    ax.set_ylim(df["avg_temperature"].min() - 1, df["avg_temperature"].max() + 1)
    fig.suptitle("Temperatura vs Tiempo", fontsize=13, fontweight="bold", x=0.08, ha="left")
    
    ax.set_title(" Variación de temperatura promedio (2020–2024)\nTendencia mensual global — promedio de todos los países\n¿Cómo ha cambiado la temperatura a lo largo del tiempo?",
                 fontsize=9, color="#555", loc="left", pad=4)

    ax.fill_between(monthly["date"], monthly["avg_temperature"], alpha=0.12, color="#4e79a7")
    ax.plot(monthly["date"], monthly["avg_temperature"], color="#1f3b73", linewidth=2.5, marker="o",
            markersize=3.5, markerfacecolor="#1f3b73", markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xlabel("Tiempo", fontsize= 9)
    ax.set_ylabel("Temperatura (°C)", fontsize= 9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}°"))
    ax.grid(color="#b1acac", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.tick_params(axis="x", rotation=20, labelsize=8)

    analysis = (
        "Análisis \nEsta gráfica muestra cómo cambia la temperatura promedio a lo largo del tiempo. "
        "Se puede observar un patrón muy claro y repetitivo, donde la temperatura \nsube y baja de forma "
        "constante cada año. Los picos representan los momentos de mayor temperatura, mientras que los "
        "puntos  más bajos corresponden \na las épocas más frías. Este comportamiento se repite de manera "
        "similar en todos los años, lo que indica que hay una tendencia estacional bastante marcada."
    )
    fig.text(0.05, -0.12, analysis, fontsize=9, color="#555", wrap=True,
             ha="left", va="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("chart1_temperatura.png", dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    print(" chart1_temperatura.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 2 — CO₂ promedio por país
# ══════════════════════════════════════════════════════════════════════════════
def chart2(df):
    co2 = (
        df.groupby("country")["co2_emission"]
        .mean()
        .reset_index()
        .rename(columns={"co2_emission": "val"})
        .sort_values("val", ascending=False)
    )

    norm = plt.Normalize(co2["val"].min(), co2["val"].max())
    colors = plt.cm.Blues(0.25 + norm(co2["val"]) * 0.65)


    fig, ax = plt.subplots() 
    apply_chart3_style(fig, ax)
    ax.set_ylim(0, co2["val"].max() * 1.2)
    fig.suptitle("Emisiones de CO₂ por país", fontsize=13, fontweight="bold", x=0.08, ha="left")
    ax.set_title("Promedio diario de emisiones 2020–2024 — ordenado de mayor a menor\n¿Qué países contaminan más?",
                 fontsize=9, color="#555", loc="left", pad=4)

    bars = ax.bar(co2["country"], co2["val"], color=colors, width=0.65)
    for bar, val in zip(bars, co2["val"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8, color="#555", fontweight="600")

    ax.set_ylabel("CO₂ promedio (ton/día)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.tick_params(axis="x", rotation=22, labelsize=8)
    ax.grid(axis="y", color="#ddd", linewidth=0.6, alpha=0.6)

    analysis = (
        "\n \nAnálisis:\nLas emisiones de CO₂ entre los países no son muy diferentes entre sí, ya que la mayoría "
        "se mantiene en un rango bastante parecido. Australia \n es el país con el valor más alto, mientras que "
        "Turquía tiene el más bajo. Aunque hay un orden de mayor a menor, la diferencia entre los países \n no es "
        "muy grande, lo que indica que todos tienen niveles de emisiones bastante similares dentro del conjunto de datos."
    )
    fig.text(0.08, -0.12, analysis, fontsize=9, color="#555", ha="left", va="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("chart2_co2.png", dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    print("✔ chart2_co2.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 3 — Distribución de consumo energético (box plot)
# ══════════════════════════════════════════════════════════════════════════════
def chart3(df):
    countries = df["country"].unique()
    groups = [df[df["country"] == c]["energy_consumption"].dropna().values for c in countries]
    medians = [np.median(g) for g in groups]
    order   = np.argsort(medians)[::-1]
    countries_sorted = [countries[i] for i in order]
    groups_sorted    = [groups[i]    for i in order]

    norm   = plt.Normalize(min(medians), max(medians))
    colors = [plt.cm.Blues(0.25 + norm(np.median(g)) * 0.6) for g in groups_sorted]

    fig, ax = plt.subplots()
    apply_chart3_style(fig, ax)  
    ax.set_ylim(2000, 16000)
    ax.set_yticks(np.arange(2000, 17000, 2000))
    fig.suptitle("Distribución del consumo energético por país", fontsize=13, fontweight="bold", x=0.08, ha="left")
    ax.set_title("Mediana, rango intercuartílico y valores extremos — ordenado por consumo mediano\n¿Cómo varía el consumo energético entre países?",
                 fontsize=9, color="#555", loc="left", pad=4)

    bp = ax.boxplot(groups_sorted, labels=countries_sorted, patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#bbb", linewidth=1.2),
                    capprops=dict(color="#bbb", linewidth=1.2),
                    flierprops=dict(marker="o", markerfacecolor="none", markersize=3, alpha=0.6))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    for i, (g, c) in enumerate(zip(groups_sorted, colors)):
        med = np.median(g)
        ax.text(i + 1, med + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                f"{med/1000:.1f}k", ha="center", fontsize=8, color="#3a3939", fontweight="700")

    ax.set_ylabel("Consumo energético (MWh)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.tick_params(axis="x", rotation=22, labelsize=8)
    ax.grid(axis="y", color="#b1acac", linestyle="--", linewidth=0.6, alpha=0.6)

    analysis = (
        "\n \n Análisis:\n Esta gráfica muestra cómo se distribuye el consumo energético en diferentes países. "
        "La mayoría tienen valores bastante parecidos,\n con medianas cercanas a los 7k. Algunos países tienen "
        "mayor variación en sus datos, lo que indica que su consumo cambia más con el tiempo,\n mientras que otros "
        "son más estables. También se observan valores más altos o más bajos de lo normal, lo que puede deberse "
        "a cambios \nen la actividad industrial o en el uso de energía."
    )
    fig.text(0.08, -0.12, analysis, fontsize=9, color="#555", ha="left", va="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("chart3_consumo.png", dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    print(" chart3_consumo.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 4 — Top 10 países por energía renovable
# ══════════════════════════════════════════════════════════════════════════════
def chart4(df):
    renew = (
        df.groupby("country")["renewable_share"]
        .mean()
        .reset_index()
        .rename(columns={"renewable_share": "val"})
        .sort_values("val", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots()
    apply_chart3_style(fig, ax)
    ax.set_xlim(0, renew["val"].max() * 1.2)
    fig.suptitle("Participación de energías renovables por país", fontsize=13, fontweight="bold", x=0.08, ha="left")
    ax.set_title("Porcentaje promedio de energía renovable sobre el total (2020–2024)\n¿Cuáles son los 10 países que utilizan más energía renovable?",
                 fontsize=9, color="#555", loc="left", pad=4)

    bars = ax.barh(renew["country"][::-1], renew["val"][::-1], color="#2a9d8f", alpha=0.9)
    for bar, val in zip(bars, renew["val"][::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, color="#444", fontweight="600")

    ax.set_xlabel("Participación renovable (%)", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", color="#ddd", linewidth=0.6, alpha=0.4)
    ax.set_facecolor("#fafafa")

    analysis = (
        "\nAnálisis:\nLa gráfica muestra los 10 países con mayor participación de energía renovable. "
        "Todos los países tienen valores muy\ncercanos entre sí (≈15.9%–16.1%), lo que muestra que no hay "
        "una diferencia muy marcada entre ellos. México y Reino Unido aparecen  \npor encima del resto, "
        "aunque la diferencia es mínima, indicando un avance relativamente equilibrado entre estos países."
    )
    fig.text(0.08, -0.12, analysis, fontsize=9, color="#555", ha="left", va="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("chart4_renovable.png", dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    print(" chart4_renovable.png guardado")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICA 6 — Energía renovable vs Emisiones CO₂ (tendencia agrupada)
# ══════════════════════════════════════════════════════════════════════════════
def chart6(df):
    df_sorted = df.sort_values("renewable_share")
    bins = pd.cut(df_sorted["renewable_share"], bins=12)
    trend = (
        df_sorted.groupby(bins, observed=True)
        .agg(x=("renewable_share", "mean"), y=("co2_emission", "mean"))
        .dropna()
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots()
    apply_chart3_style(fig, ax) 
    ax.set_xlim(5, 35)
    ax.set_ylim(400, 460)
    fig.suptitle("Relación entre energía renovable y emisiones de CO₂",
                 fontsize=13, fontweight="bold", x=0.08, ha="left")
    ax.set_title("Promedios agrupados — tendencia general\n¿Más energía renovable reduce las emisiones? \n",
                 fontsize=9, color="#555", loc="left", pad=4)

    ax.fill_between(trend["x"], trend["y"], alpha=0.1, color="#2a9d8f")
    ax.plot(trend["x"], trend["y"], color="#1b7f6b", linewidth=2.5, marker="o",
            markersize=4, markerfacecolor="#1b7f6b", alpha=0.85)

    ax.set_xlim(5, 35)
    ax.set_ylim(400, 460)
    ax.set_xlabel("Energía renovable (%)", fontsize=9)
    ax.set_ylabel("Emisiones de CO₂", fontsize=9)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(color="#b1acac", linestyle="--", linewidth=0.6, alpha=0.6)

    analysis = (
        "\n Análisis:\nEsta gráfica muestra cómo las emisiones se mantienen relativamente estables a lo largo "
        "de los distintos niveles de energía renovable, sin cambios muy bruscos.\nNo se evidencia una disminución "
        "clara de las emisiones a medida que aumenta el uso de energías renovables. En algunos \npuntos incluso se "
        "presentan ligeros aumentos. Al final de la gráfica se observa una caída más marcada en las emisiones, "
        "lo que podría indicar un posible\nefecto positivo del aumento en el uso de energías renovables."
    )
    fig.text(0.08, -0.12, analysis, fontsize=9, color="#555", ha="left", va="top", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("chart6_renovable_co2.png", dpi=150, bbox_inches="tight")
    st.pyplot(fig)
    print("✔ chart6_renovable_co2.png guardado")


# ── Ejecutar todas las gráficas ─────────────────────────────────────────────
if __name__ == "__main__":
    chart1(df)
    chart2(df)
    chart3(df)
    chart4(df)
    chart6(df)