#!/bin/zsh
set -euo pipefail

target=work/google_carfac
commit=c74663cc7d05713ae2f2308765eb040530a81c7f
if [[ ! -d "$target/.git" ]]; then
  git clone https://github.com/google/carfac.git "$target"
fi
git -C "$target" fetch origin "$commit"
git -C "$target" checkout --detach "$commit"
test "$(git -C "$target" rev-parse HEAD)" = "$commit"
echo CARFAC_PINNED_SOURCE_OK
