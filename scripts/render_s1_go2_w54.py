#!/usr/bin/env python3
"""Full recorded replay using W54 compact records, without new physics."""
import render_s1_go2_r12 as renderer
renderer.SMALL=renderer.SMALL/'w54'
if __name__=='__main__':renderer.main()
