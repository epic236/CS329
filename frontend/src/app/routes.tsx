import { createBrowserRouter } from "react-router";
import { TranslatorInput } from "./components/TranslatorInput";
import { TranslatorResult } from "./components/TranslatorResult";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: TranslatorInput,
  },
  {
    path: "/result",
    Component: TranslatorResult,
  },
]);
