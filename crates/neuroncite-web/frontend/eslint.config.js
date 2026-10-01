// ESLint flat config for the SolidJS frontend.
//
// Replaces the former .eslintrc.json: eslint-plugin-solid 0.18 only ships
// flat-config presets, and ESLint 10 drops the eslintrc format entirely.
// The rule set is the same as before: eslint:recommended,
// @typescript-eslint/recommended, solid/typescript, jsx-a11y/recommended,
// plus the project-specific overrides at the bottom.

import js from "@eslint/js";
import tsPlugin from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import jsxA11y from "eslint-plugin-jsx-a11y";
import solid from "eslint-plugin-solid/configs/typescript";

export default [
  { ignores: ["dist/**", "node_modules/**"] },
  js.configs.recommended,
  ...tsPlugin.configs["flat/recommended"],
  solid,
  jsxA11y.flatConfigs.recommended,
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        project: "./tsconfig.json",
        ecmaVersion: 2022,
        sourceType: "module",
      },
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/no-unused-vars": ["error", { argsIgnorePattern: "^_" }],
      "solid/reactivity": "warn",
      "jsx-a11y/no-noninteractive-element-interactions": "warn",
    },
  },
];
