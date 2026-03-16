/**
 * Path normalization utilities for Windows extended-length path prefixes.
 *
 * Windows file system APIs sometimes return paths with extended-length
 * prefixes like "\\?\", "\\.\", or "\\?\UNC\" for paths exceeding the
 * MAX_PATH limit of 260 characters. These prefixes bypass the Win32 path
 * parser and pass the path directly to the NT object manager. The prefixes
 * are not meaningful for display or comparison purposes and should be
 * stripped before presenting paths to the user or comparing them.
 *
 * Reference: https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file
 */

/**
 * Regex matching Windows extended-length path prefixes at the start of a string.
 *
 * Matches three prefix patterns:
 *   - \\?\UNC\  -- Extended UNC prefix for network paths. After stripping,
 *     the path should begin with \\ to remain a valid UNC path.
 *   - \\?\      -- Extended local path prefix (e.g., \\?\C:\folder).
 *   - \\.\      -- Device path prefix (e.g., \\.\PhysicalDrive0).
 *
 * The UNC pattern is listed first because it is a longer prefix that starts
 * with the same characters as the generic \\?\ prefix.
 */
const WINDOWS_PREFIX_RE = /^(?:\\\\\?\\UNC\\|\\\\\?\\|\\\\\.\\)/;

/**
 * Strips Windows extended-length path prefixes from a file system path.
 * Returns the path unchanged if it does not start with a recognized prefix.
 *
 * For UNC paths (\\?\UNC\server\share), the result is \\server\share because
 * the UNC prefix replacement drops \\?\UNC\ and prepends \\ to preserve
 * valid UNC path syntax.
 *
 * @param path - The file system path to normalize.
 * @returns The path with any extended-length prefix removed.
 */
export function normalizeWindowsPath(path: string): string {
  if (path.startsWith("\\\\?\\UNC\\")) {
    return "\\\\" + path.slice(8);
  }
  return path.replace(WINDOWS_PREFIX_RE, "");
}
