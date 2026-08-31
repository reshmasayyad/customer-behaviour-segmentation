# Publishing to GitHub

The ZIP includes the local `.git` directory and its commit history. Extract the complete repository folder, including that directory. Do not run `git init` again and do not upload only individual files through the GitHub website if you want to preserve the commit history.

Create an empty GitHub repository named `customer-behaviour-segmentation`. Do not initialize the remote with a README, license or `.gitignore`, because those already exist locally.

From the extracted folder:

```bash
git status
git log --oneline --reverse
git config user.name "Your actual name"
git config user.email "Your verified GitHub email or GitHub noreply address"
git remote add origin https://github.com/YOUR_USERNAME/customer-behaviour-segmentation.git
git push -u origin main
```

Replace the name, email and URL placeholders. Authenticate through your normal Git credential manager or GitHub CLI; do not add credentials to project files. Local contributor settings apply to future commits.

If the folder already has a remote, inspect it with `git remote -v` and update the intended URL with `git remote set-url origin URL`. Push to an empty repository to avoid unrelated-history conflicts.

The separately supplied Git bundle is an alternative portable copy of the history. If a ZIP extractor omits `.git`, clone the bundle instead:

```bash
git clone customer-segmentation.bundle customer-behaviour-segmentation
```

Then configure the GitHub URL with `git remote set-url origin https://github.com/YOUR_USERNAME/customer-behaviour-segmentation.git` and push the `main` branch.
