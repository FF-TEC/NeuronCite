/**
 * Accessibility utilities for making non-semantic elements interactive.
 *
 * When a <div> or <span> is used as a clickable element instead of a
 * <button>, it lacks keyboard accessibility and screen reader semantics.
 * The clickableProps function returns a set of HTML attributes that make
 * the element behave like a button: role="button" for screen readers,
 * tabIndex=0 for keyboard focus, and onKeyDown handling for Enter and
 * Space keys to trigger the click action.
 *
 * Usage: spread the returned object onto the element:
 *   <div {...clickableProps(() => handleClick())} />
 */

import type { JSX } from "solid-js";

/**
 * Properties returned by clickableProps that should be spread onto a
 * non-semantic clickable element to make it keyboard-accessible and
 * screen-reader-friendly.
 */
interface ClickableAttributes {
  role: "button";
  tabIndex: number;
  class: string;
  onKeyDown: JSX.EventHandler<HTMLElement, KeyboardEvent>;
}

/**
 * Returns HTML attributes that add button semantics to a non-semantic element.
 *
 * The returned object contains:
 *   - role="button" -- announces the element as an interactive button to
 *     screen readers via the ARIA role attribute.
 *   - tabIndex=0 -- includes the element in the natural tab order so
 *     keyboard users can focus it.
 *   - class="clickable-focus" -- applies the focus ring style defined in
 *     components.css, providing a visible keyboard focus indicator that
 *     satisfies WCAG 2.1 SC 2.4.7 (Focus Visible).
 *   - onKeyDown -- calls onClick when Enter or Space is pressed, matching
 *     the native <button> keyboard interaction model. Space key events
 *     are prevented from scrolling the page via preventDefault().
 *
 * @param onClick - The function to invoke when the element is activated
 *   by keyboard (Enter or Space). This should be the same function passed
 *   to the element's onClick handler.
 * @param baseClass - Optional existing CSS class string to merge with the
 *   clickable-focus class. Use this when the element already has its own
 *   class attribute so that spreading this object does not duplicate the
 *   class prop (TypeScript TS2783). The result is "baseClass clickable-focus".
 */
export function clickableProps(
  onClick: () => void,
  baseClass?: string,
): ClickableAttributes {
  return {
    role: "button",
    tabIndex: 0,
    class: baseClass ? `${baseClass} clickable-focus` : "clickable-focus",
    onKeyDown: (e: KeyboardEvent) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        onClick();
      }
    },
  };
}
