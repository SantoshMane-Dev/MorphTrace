<script>
  import { T, useTask } from "@threlte/core";

  let group;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useTask((delta) => {
    if (group && !reducedMotion) {
      group.rotation.y += delta * 0.16;
      group.rotation.x = Math.sin(performance.now() * 0.00035) * 0.12;
    }
  });
</script>

<T.AmbientLight intensity={1.6} />
<T.PointLight position={[3, 3, 4]} intensity={18} color="#f2a7b5" />
<T.Group bind:ref={group}>
  <T.Mesh>
    <T.IcosahedronGeometry args={[0.86, 1]} />
    <T.MeshStandardMaterial color="#315c4c" roughness={0.28} metalness={0.18} transparent opacity={0.38} />
  </T.Mesh>
  <T.Mesh>
    <T.IcosahedronGeometry args={[1.12, 2]} />
    <T.MeshStandardMaterial
      color="#f2a7b5"
      emissive="#f2a7b5"
      emissiveIntensity={0.25}
      roughness={0.32}
      metalness={0.15}
      wireframe
    />
  </T.Mesh>
  <T.Mesh rotation={[0.35, 0.2, 0.45]}>
    <T.TorusGeometry args={[1.42, 0.02, 8, 120]} />
    <T.MeshBasicMaterial color="#f2a7b5" />
  </T.Mesh>
  <T.Mesh rotation={[1.15, 0.7, 0.1]}>
    <T.TorusGeometry args={[1.52, 0.014, 8, 120]} />
    <T.MeshBasicMaterial color="#fff7e8" transparent opacity={0.82} />
  </T.Mesh>
  <T.Mesh position={[0.72, 0.56, 0.44]}>
    <T.SphereGeometry args={[0.075, 20, 20]} />
    <T.MeshBasicMaterial color="#f2a7b5" />
  </T.Mesh>
</T.Group>
