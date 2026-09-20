import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import { Shell } from "./components/layout/Shell";
import { ChartSkeleton, EmptyState } from "./components/ui/States";
import { Compass } from "lucide-react";

import Overview from "./pages/Overview";
import Forecast from "./pages/Forecast";

// Heavier / less-visited pages are split into their own chunks.
const ModelPerformance = lazy(() => import("./pages/ModelPerformance"));
const FeatureInsights = lazy(() => import("./pages/FeatureInsights"));
const About = lazy(() => import("./pages/About"));

export default function App() {
  return (
    <Suspense fallback={<div className="page"><ChartSkeleton height={360} /></div>}>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Overview />} />
          <Route path="forecast" element={<Forecast />} />
          <Route path="performance" element={<ModelPerformance />} />
          <Route path="insights" element={<FeatureInsights />} />
          <Route path="about" element={<About />} />
          <Route path="*" element={<div className="page"><EmptyState icon={<Compass size={20} />} title="Page not found" message="Use the navigation to pick a section." /></div>} />
        </Route>
      </Routes>
    </Suspense>
  );
}
