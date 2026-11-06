<script>
  import { Canvas } from "@threlte/core";
  import ForensicModel from "./ForensicModel.svelte";
</script>

<Canvas dpr={[1, 1.5]} camera={{ position: [0, 0, 5.2], fov: 38 }} scene={{ background: null }}>
  <ForensicModel />
</Canvas>
