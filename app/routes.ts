import { type RouteConfig, route } from "@react-router/dev/routes";

export default [
  route("/", "./routes/home.tsx"),
  route("/mobile", "./routes/mobile.tsx"),
] satisfies RouteConfig;