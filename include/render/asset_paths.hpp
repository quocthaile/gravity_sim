#pragma once

#include <filesystem>
#include <string>

namespace grav_render
{

// Optional explicit asset root (e.g., from CLI --asset-root or unit tests).
void SetAssetRootOverride(const std::filesystem::path &rootPath);
void ClearAssetRootOverride();

// Directory containing the running executable on Windows, or current_path() fallback.
std::filesystem::path GetExecutableDirectory();

// Resolves a relative asset path (e.g. "shaders/grid.comp") against:
// 1. Explicit override root (if set; when override is active and strict, checks only override)
// 2. Executable directory
// 3. Parent of executable directory (for multi-config build layouts)
// 4. Current working directory
// 5. Parent of current working directory
// Throws std::runtime_error with attempted locations if the file does not exist.
std::filesystem::path ResolveAssetPath(const std::string &relativePath);

// Non-throwing existence probe using the same search order.
bool TryResolveAssetPath(const std::string &relativePath, std::filesystem::path &outResolved);

} // namespace grav_render

