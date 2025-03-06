import type { Route } from "./+types/home";
import { test } from "../welcome/test";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "AI Colorizer" },
    { name: "description", content: "AI Colorizer test" },
  ];
}

export default function Home() {
  return test();
}
