import { mount } from "svelte";
import ForensicObject from "./ForensicObject.svelte";
import "./visuals.css";

document.querySelectorAll("[data-morphtrace-scene]").forEach((target) => {
  mount(ForensicObject, { target });
});
