using BlackHoles
# example: BlackHoles.compute_waveform(dt, sol; coorbital=false)


Commit each file.

---

# 3) GitHub Actions to build & publish

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation
on:
  push:
    branches: [ main ]   # or master
  pull_request:
    branches: [ main ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1'
      - name: Instantiate package
        run: julia --project -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
      - name: Instantiate docs env
        run: julia --project=docs -e 'using Pkg; Pkg.instantiate()'
      - name: Build & deploy docs
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: julia --project=docs docs/make.jl
