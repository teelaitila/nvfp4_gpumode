## Optimization ideas to try

- Increase CTA count by reducing tile size to reach >= 1 wave.
- Tune `num_ab_stage`/`num_c_stage` for better occupancy and lower SMEM.
- DONE Add cluster + TMA multicast for A/SFA across N tiles (working with `cluster_shape_mn=(1,4)`).
- Implement 2-CTA MMA path (cta_group TWO) and benchmark M=512 shapes.
- Try direct R2G epilogue store vs TMA store (small grids).
- Revisit prefetch distance vs stage count for these K sizes.
