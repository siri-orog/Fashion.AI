def suggest_outfits(skin_tone, dominant_color):
    suggestions = []

    # Basic skin tone logic
    if skin_tone == "fair":
        suggestions.append("Try pastel shades like lavender, mint, or baby blue.")
    elif skin_tone == "medium":
        suggestions.append("Go for warm tones like coral, teal, or mustard.")
    else:
        suggestions.append("Earth tones like olive, maroon, and beige look great!")

    # Complementary color idea
    if dominant_color:
        r, g, b = dominant_color
        if r > g and r > b:
            suggestions.append("Avoid bright reds; try soft blues or greys.")
        elif g > r and g > b:
            suggestions.append("Neutral tones and whites enhance your look.")
        else:
            suggestions.append("Go for warm accessories like gold or tan.")

    suggestions.append("Add accessories to balance the tone — e.g., silver for cool, gold for warm palettes.")
    return suggestions
