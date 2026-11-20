# Demo

Two parts, each deployable on its own:

| Folder | What it is | Where it runs |
| :-- | :-- | :-- |
| `site/` | the website: live demo, samples beside Vāgdhenu's, how it was made, how to run it | any static host; GitHub Pages via `.github/workflows/pages.yml` |
| `space/` | the live demo: a Gradio app that chants a verse in Nandinī's voice, Vāgdhenu's, or both | a Hugging Face Space (ZeroGPU), or any machine with `python app.py` |

**The website** is plain HTML with its audio beside it, so it works from any folder or host. It embeds the live demo
from the Space URL set at the bottom of `site/index.html` (`DEMO`), and shows a note if the Space can't be reached.
To publish it with GitHub Pages, set the repository's Pages source to "GitHub Actions"; the workflow then deploys
`demo/site/` on every push that changes it.

**The live demo** is described in `space/README.md`, which is also the Space's card:

```bash
cd demo/space && huggingface-cli upload <user>/nandini-demo . --repo-type space
```
