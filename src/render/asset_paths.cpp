#include "render/asset_paths.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace grav_render
{
namespace
{
std::filesystem::path g_assetRootOverride;
bool g_hasAssetRootOverride = false;

std::vector<std::filesystem::path> CandidateRoots()
{
    std::vector<std::filesystem::path> roots;
    if (g_hasAssetRootOverride)
    {
        roots.push_back(g_assetRootOverride);
        return roots;
    }

    const std::filesystem::path exeDir = GetExecutableDirectory();
    if (!exeDir.empty())
    {
        roots.push_back(exeDir);
        if (exeDir.has_parent_path())
        {
            roots.push_back(exeDir.parent_path());
            if (exeDir.parent_path().has_parent_path())
            {
                roots.push_back(exeDir.parent_path().parent_path());
            }
        }
    }

    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    if (!ec && !cwd.empty())
    {
        roots.push_back(cwd);
        if (cwd.has_parent_path())
        {
            roots.push_back(cwd.parent_path());
        }
    }
    return roots;
}
} // namespace

void SetAssetRootOverride(const std::filesystem::path &rootPath)
{
    g_assetRootOverride = rootPath;
    g_hasAssetRootOverride = true;
}

void ClearAssetRootOverride()
{
    g_assetRootOverride.clear();
    g_hasAssetRootOverride = false;
}

std::filesystem::path GetExecutableDirectory()
{
#ifdef _WIN32
    wchar_t buffer[4096] = {};
    const DWORD length = GetModuleFileNameW(nullptr, buffer, static_cast<DWORD>(std::size(buffer)));
    if (length > 0 && length < std::size(buffer))
    {
        return std::filesystem::path(buffer).parent_path();
    }
#endif
    std::error_code ec;
    const std::filesystem::path cwd = std::filesystem::current_path(ec);
    return ec ? std::filesystem::path{} : cwd;
}

bool TryResolveAssetPath(const std::string &relativePath, std::filesystem::path &outResolved)
{
    if (relativePath.empty())
    {
        return false;
    }

    std::error_code ec;
    const std::filesystem::path rel(relativePath);
    if (rel.is_absolute())
    {
        if (std::filesystem::is_regular_file(rel, ec))
        {
            outResolved = rel;
            return true;
        }
        return false;
    }

    for (const auto &root : CandidateRoots())
    {
        const std::filesystem::path candidate = root / rel;
        if (std::filesystem::is_regular_file(candidate, ec))
        {
            outResolved = std::filesystem::weakly_canonical(candidate, ec);
            if (ec)
            {
                outResolved = candidate;
            }
            return true;
        }
    }
    return false;
}

std::filesystem::path ResolveAssetPath(const std::string &relativePath)
{
    std::filesystem::path resolved;
    if (TryResolveAssetPath(relativePath, resolved))
    {
        return resolved;
    }

    std::ostringstream oss;
    oss << "Unable to resolve asset '" << relativePath << "'. Checked roots:";
    for (const auto &root : CandidateRoots())
    {
        oss << " [" << root.string() << "]";
    }
    throw std::runtime_error(oss.str());
}

} // namespace grav_render

