def map_strings_to_colors(strings):
    colors = [
        "#186B66",
        "#B97A6A",
        "#66014A",
        "#085D8F",
        "#065D6F",
        "#7C3F40",
        "#3C5B5C",
        "#D26D7E",
        "#4A2B73",
        "#4a536b",
    ]


    i = 0
    while len(colors) < len(strings):
        colors.append(colors[i])
        i += 1

    return {string: colors[i] for i, string in enumerate(strings)}