# Paper Narrative Review: What Story Is `paper.tex` Telling?

*Full read-through of `Context/AGNPaper/paper.tex` against its figures, checking whether the argument holds together end to end. Generated 2026-08-06, updated same day after restructuring the CIGALE Results subsections into four narrative beats. §4/§5 refreshed 2026-09-25 — the paper is now a complete draft and most gaps listed below have since been written; see `docs/paper_next_steps.md` for the completion status.*

## 1. The core story, as the paper currently argues it

> AGN light contaminates the observed colours of their host galaxies. For **unobscured (Type 1)** AGN, this contamination is strong enough in rest-frame UVJ colour space to make quiescent galaxies look star-forming - a real source of bias for any survey using UVJ to select quiescent populations. **Obscured (Type 2)** AGN barely touch UVJ but show up strongly in **mid-infrared (IRAC)** colours instead, so the two diagnostics are sensitive to complementary AGN populations. This isn't just a model artefact: real ZFOURGE galaxies, decomposed with CIGALE to isolate and remove their fitted AGN component, show the same signature - hosts move back toward quiescent once the AGN light is subtracted, the effect scales with how much AGN light was there, and it's redshift-dependent in a way that's consistent with the physical picture.

Structure: model → apply the model → confirm on real data two different ways (ZFOURGE synthetic injection, then CIGALE real decomposition). The CIGALE half of that argument is now broken into four narrative beats, one per Results subsection (see §2).

## 2. Section-by-section: claim vs. figure support

| Section | Claim | Figure(s) | Verdict |
|---|---|---|---|
| §4.1 IRAC validation | Type 1 composites drop out of the Lacy wedge (completeness 0.26→0.06); Type 2 composites move *into* it (→98% at 70% contribution) | Fig 1 (`Brown-IRAC-combined`), Table 1 | **Solid.** |
| §4.2 UVJ theoretical evolution | Type 1 shifts host colours bluer, into the star-forming region; Type 2 barely moves | Fig 2 (`Brown-UVJ-combined`), Fig 3 (`uvj-fractions-combined`), Fig 4 (`composite_uvj`) | **Solid.** |
| §4.3 ZFOURGE synthetic injection | Real ZFOURGE galaxies, with the theoretical Type 1 model injected on top, show the same star-forming-ward migration | Fig 5 (`ZFOURGE-EAZY-UVJ`), Fig 6 (`ZFOURGE-EAZY-UVJSingleGalaxy_Comparison`) | **Solid.** Bridge from "the model does this" to "real galaxy colours do this too." |
| §4.4 **The Systematic UVJ Colour Shift** (new) | The whole AGN-host population's colours shift with fracAGN, not just galaxies that cross a class boundary | Fig 7 (`cigale-allhosts-fracagn-vectors`) | **Solid** — lead-in paragraph now written (2026-09-25 refresh). |
| §4.5 **Redshift Dependence of the Colour Shift** (new) | The shift magnitude and mean fracAGN both rise with redshift | Fig 8 (`cigale-allhosts-redshift-grid`) | **Solid** — redshift-dependence discussion (confound vs genuine effect) now written. |
| §4.6 **UVJ Offset and Migration to the Quiescent Region** (new) | Vector offset rises monotonically with fracAGN (ρ=0.49); migration rate into quiescent is hump-shaped, peaking fracAGN≈0.1-0.5 | Fig 9 (`fracagn-distribution-offset`) | **Solid, has a paragraph already** (the pre-existing "SED decomposition via CIGALE provides..." text). |
| §4.7 **Redshift Dependence of the Quiescent Migration** (new) | Migration hump isn't solely a redshift-dilution artefact (survives within z<1.5); recovery peaks z=0.5-1.0, null above z≈1.5 | Fig 10 (`fracagn-redshift-confound`), Fig 11 (`hidden-quiescent-redshift`) | **Solid, statistically rigorous** (Spearman tests, bootstrap CIs, AIC model comparisons) — Fig 11 paragraph now written. |
| Discussion | VJ shift deviation attributed to missing intermediate-type AGN; dusty fraction falls after decomposition, consistent with cosmic SFH | Refs back to Fig 4, Fig 7 | **Solid**, reads fine as-is - this is the intended home for CIGALE synthesis/interpretation (see §3 below), not a new subsection inside Results. |
| Conclusions | Theoretical predictions → empirical ZFOURGE/CIGALE validation → cosmic time peak summary | - | **Solid** — written (2026-09-25 refresh). |

## 3. Restructuring done this pass, and why

Your narrative (CIGALE method → systematic shift → its redshift dependence → UVJ offset/migration → its redshift dependence) mapped cleanly onto:

- **New Methodology subsection**, `\subsection{SED Decomposition via CIGALE}` (§3.4, `subsec:cigale_method`) - grounds the `fracAGN` terminology and "decomposition" concept that Results throws at the reader with zero explanation otherwise. Todo comment has the factual content to write from.
- **Results split from 2 subsections into 4**, mirroring "shift → its redshift dependence" twice: `subsec:cigale_systematic_shift` (Fig 7) → `subsec:cigale_redshift_shift` (Fig 8) → `subsec:cigale_offset_migration` (Fig 9) → `subsec:cigale_migration_redshift` (Figs 10-11). Fig 10 was kept with the migration-redshift pairing rather than the offset pairing, since its entire purpose is checking whether the migration hump is a redshift artifact.
- **No new "discussion" subsection inside Results** - recommended keeping the existing `subsec:disc_comparison` in the Discussion section as the home for CIGALE synthesis, since it already does that job for the theoretical/observational comparison and splitting the paper's Results/Discussion separation only for the CIGALE portion would be inconsistent with the rest of the structure.

All changes verified: figure count/order unchanged (still 11), no duplicate labels or dangling refs, and the whole edited region (Methodology addition + 4-subsection Results split) isolate-compiles cleanly.

## 4. What's actually missing or broken in the connective tissue

*(Updated 2026-09-25 — items 1, 2, 3, and 6's Conclusions clause have since been resolved; struck through below and kept for history. Items 4 and 5 are still open.)*

1. ~~**No lead-in paragraph for §4.4/§4.5** (Figs 7-8)~~ — written.
2. ~~**§4.5's "why does the shift grow with redshift" discussion doesn't exist yet**~~ — written.
3. ~~**§4.7's Fig 11 still has no paragraph**~~ — written.
4. **The theory tie-back argument has no home** - a figure that overlaid observed CIGALE decomposition arrows on the theoretical Type 1 tracks was removed as redundant with Fig 7 a few turns back; the claim it made ("decomposition retraces the contamination path in reverse") doesn't live anywhere now. Still an open decision — not confirmed resolved.
5. **Figure 7's fate** - still open whether to reconsider merging it into Fig 8, flagged in an earlier pass and never resolved either way.
6. ~~**Conclusions section is empty.**~~ — written. **End-matter** (Funding, Acknowledgments, Data Availability, Competing Interests) is filled in; Author Contributions is intentionally still a placeholder; Ethical Standards is already accurate boilerplate.

## 5. Recommended next steps, in order

*(Updated 2026-09-25: steps 1, 2, 3, and 6 are done. Remaining open items:)*

1. Decide on the theory-tie-back claim (§4 item 4 above): revive as a Discussion sentence, or drop deliberately.
2. Decide on Figure 7's fate if you still want to reconsider merging it into Fig 8 (§4 item 5).
3. Fill in Author Contributions when ready.
4. Re-read the 14 `% AI-DRAFTED`/`% AI-UPDATED` inline comments throughout the paper in your own voice and verify against fresh notebook runs before submission (see `docs/paper_next_steps.md`).
