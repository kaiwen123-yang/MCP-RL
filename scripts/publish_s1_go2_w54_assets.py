#!/usr/bin/env python3
"""Reuse the established server transport with the isolated W54 ledger."""
import publish_s1_go2_r12 as transport
transport.SMALL=transport.SMALL/'w54'
if __name__=='__main__':transport.main()
