/**
 * URL sanitization utilities for user-supplied URLs (BibTeX entries,
 * external links). Prevents rendering of dangerous URI schemes like
 * javascript:, data:, and vbscript: as clickable href attributes.
 *
 * Safe schemes: http, https, ftp, and the doi: pseudo-scheme used
 * in academic BibTeX entries. All other schemes are treated as unsafe.
 */

/** Set of URI schemes that are safe to render as clickable links. */
const SAFE_SCHEMES = new Set(["http:", "https:", "ftp:", "ftps:"]);

/**
 * Regular expression for validating DOI identifiers. A DOI must start with
 * the registrant prefix "10." followed by four or more digits (the registrant
 * code), a forward slash, and one or more characters that are not whitespace,
 * double-quote, angle brackets, or other characters reserved for HTML/XML.
 * This rejects bare "10." strings and partial DOIs that would produce
 * invalid https://doi.org/ redirect targets.
 */
const DOI_PATTERN = /^10\.[0-9]{4,}\/[^\s"<>]+$/;

/**
 * Returns true when the given string is a syntactically valid DOI identifier.
 * A valid DOI starts with "10." followed by a numeric registrant code of at
 * least four digits, a forward slash, and a non-empty suffix that contains
 * no whitespace or HTML-reserved characters.
 *
 * This function does not verify that the DOI is registered or resolvable;
 * it only validates the structural format.
 *
 * @param doi - The candidate DOI string (without the doi: scheme prefix)
 */
export function isSafeDoiIdentifier(doi: string): boolean {
  return DOI_PATTERN.test(doi);
}

/**
 * Returns true when the given URL string uses a safe protocol scheme.
 * Returns false for javascript:, data:, vbscript:, file:, and any
 * other scheme not in the allowlist. Also returns false for empty,
 * undefined, or unparseable strings.
 *
 * DOI identifiers matching the DOI_PATTERN are treated as safe because
 * they are resolved to https://doi.org/ links by the rendering layer.
 */
export function isSafeUrl(url: string | undefined | null): boolean {
  if (!url || url.trim().length === 0) return false;

  const trimmed = url.trim();

  // DOI identifiers (e.g., "10.1000/xyz") are safe -- they are not URI
  // schemes and are rendered via https://doi.org/ prefix. Use the
  // structured regex to reject malformed "10." strings.
  if (isSafeDoiIdentifier(trimmed)) return true;

  try {
    const parsed = new URL(trimmed);
    return SAFE_SCHEMES.has(parsed.protocol);
  } catch {
    // Relative URLs or plain text without a scheme -- these cannot be
    // javascript: or data: attacks, but they are also not valid links.
    return false;
  }
}

/**
 * Sanitizes a URL for use in an href attribute. Returns the original URL
 * if it uses a safe scheme, or an empty string if it does not. This
 * prevents injection of javascript:, data:, or vbscript: URIs into the
 * DOM.
 */
export function sanitizeHref(url: string | undefined | null): string {
  return isSafeUrl(url) ? url!.trim() : "";
}
