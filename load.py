import pandas as pd
import matplotlib.colors as mcolors

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# CSS4 colors
css4_colors = mcolors.CSS4_COLORS
css4_data = [
    {"color_name": name, "r": r, "g": g, "b": b}
    for name, hex_val in css4_colors.items()
    for r, g, b in [hex_to_rgb(hex_val)]
]

# XKCD colors
xkcd_colors = mcolors.XKCD_COLORS
xkcd_data = [
    {"color_name": name.replace("xkcd:", "").replace(" ", "_"), "r": r, "g": g, "b": b}
    for name, hex_val in xkcd_colors.items()
    for r, g, b in [hex_to_rgb(hex_val)]
]

# Combine and remove duplicates
all_colors = css4_data + xkcd_data
df = pd.DataFrame(all_colors).drop_duplicates(subset="color_name")

# Save to CSV
df.to_csv("colors.csv", index=False)
