import { useApp } from "../../context/AppContext";
import { Segmented } from "../ui/Segmented";

const OPTIONS = [7, 14, 28, 42];

export function HorizonSelect() {
  const { horizon, setHorizon } = useApp();
  return (
    <Segmented ariaLabel="Forecast horizon" value={horizon} onChange={setHorizon}
      options={OPTIONS.map((d) => ({ value: d, label: `${d}d` }))} />
  );
}
