import { defineConfig } from "astro/config";

export default defineConfig({
  srcDir: "./web",
  output: "static",
  build: {
    format: "file"
  }
});
