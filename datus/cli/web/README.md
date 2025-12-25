Custom Streamlit theme for Datus Web Chatbot
==========================================

This small README explains the custom theme injected into the Streamlit Chatbot page.

Location
--------
- The custom CSS is injected from `datus/cli/web/chatbot.py` via the `CUSTOM_THEME_CSS` variable.

Enable / Disable
----------------
- The theme injection can be toggled by setting `ENABLE_CUSTOM_THEME = False` in `datus/cli/web/chatbot.py`.

What it changes
---------------
- Sidebar title sizing and weight (makes `AI Agent` more prominent).
- Main page title and caption sizing and color.
- Rounded chat input and light border styling.
- Slightly adjusted expander and session item paddings.

Notes
-----
- This is a best-effort theme using Streamlit's supported HTML injection. Streamlit updates may require adjustments to selectors in `CUSTOM_THEME_CSS`.


