#!/usr/bin/env bash
# Private node configuration stays outside Git at /root/.config/mihomo.
if ! tmux has-session -t d57-clash 2>/dev/null; then
    tmux new-session -d -s d57-clash \
        'exec /usr/local/bin/mihomo -d /root/.config/mihomo >> /root/.config/mihomo/mihomo.log 2>&1'
fi
