#!/usr/bin/env bash
# Source with academic, pc, clash, or direct. Policy: academic first; after
# three network failures of the same operation, start the PC tunnel and use pc.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
case "${1:-academic}" in
    academic)
        source /etc/network_turbo >/dev/null
        ;;
    pc)
        export http_proxy=http://127.0.0.1:17897
        export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
        ;;
    clash)
        bash "$(dirname "${BASH_SOURCE[0]}")/start_d57_clash.sh"
        export http_proxy=http://127.0.0.1:7890
        export https_proxy="$http_proxy" HTTP_PROXY="$http_proxy" HTTPS_PROXY="$http_proxy"
        ;;
    direct) ;;
    *) printf 'Usage: source scripts/autodl_network.sh academic|pc|clash|direct\n' >&2; return 2 ;;
esac
export no_proxy=localhost,127.0.0.1,::1 NO_PROXY=localhost,127.0.0.1,::1
