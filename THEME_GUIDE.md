# CyberMoE: Theme Guide

CyberMoE now supports theme customization with light and dark modes. This document explains how to use and extend the theming system.

## Available Themes

The app currently supports three theme options:
- **Light**: The default Streamlit light theme
- **Dark**: A custom dark theme optimized for readability
- **System**: Automatically follows your operating system's preference

## Selecting a Theme

You can select your preferred theme from the "💫 Appearance" section in the sidebar. The theme will persist during your session.

## Theme Components

The theme system adjusts:
- Background and text colors
- Chart and visualization palettes
- Container borders and shadows
- Code and data visualization styling

## Custom CSS Styling

The dark mode implementation uses Streamlit's custom CSS capabilities. If you need to extend the theming:

1. Locate the CSS section in `app.py` (search for "Custom CSS for theme support")
2. Add additional selectors as needed
3. Use the `[data-theme="dark"]` attribute selector for dark-mode specific styles

## Visualization Considerations

When adding new visualizations to the app:
- Avoid hard-coded colors whenever possible
- Use the built-in theme-aware color palettes
- Test your visualizations in both light and dark modes

## Extending Theme Support

To add new theme options:
1. Extend the radio button options in the sidebar
2. Add corresponding CSS rules in the custom CSS section
3. Test with various screen sizes and browsers

## Troubleshooting

If theme elements appear inconsistent:
- Try refreshing the page
- Clear your browser cache
- Check for custom CSS conflicts with browser extensions