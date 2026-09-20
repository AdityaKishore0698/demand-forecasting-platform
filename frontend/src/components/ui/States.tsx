import { AlertTriangle, Inbox, RefreshCw } from "lucide-react";
import type { CSSProperties, ReactNode } from "react";
import type { UseQueryResult } from "@tanstack/react-query";
import { ApiError } from "../../api/client";

export function Skeleton({ h = 16, w = "100%", r, className = "" }: { h?: number | string; w?: number | string; r?: number; className?: string }) {
  const style: CSSProperties = { height: h, width: w, ...(r != null ? { borderRadius: r } : {}) };
  return <div className={`skeleton ${className}`} style={style} aria-hidden />;
}

export function ChartSkeleton({ height = 280 }: { height?: number }) {
  return (
    <div style={{ padding: "6px 6px 0" }} role="status" aria-label="Loading chart">
      <Skeleton h={height} r={12} />
    </div>
  );
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const api = error instanceof ApiError ? error : null;
  const unreachable = api?.status === 0;
  const message = error instanceof Error ? error.message : "Something went wrong.";
  return (
    <div className="state" role="alert">
      <div className="state__icon state__icon--error"><AlertTriangle size={20} aria-hidden /></div>
      <h3>{unreachable ? "Can't reach the API" : "Couldn't load this data"}</h3>
      <p>{message}</p>
      {unreachable && <p>Start it with <code>make api</code> (or <code>uvicorn api.main:app</code>) and reload.</p>}
      {onRetry && (
        <button className="btn" onClick={onRetry}><RefreshCw aria-hidden /> Try again</button>
      )}
    </div>
  );
}

export function EmptyState({ title, message, icon }: { title: string; message?: string; icon?: ReactNode }) {
  return (
    <div className="state">
      <div className="state__icon">{icon ?? <Inbox size={20} aria-hidden />}</div>
      <h3>{title}</h3>
      {message && <p>{message}</p>}
    </div>
  );
}

/** Render loading / error / data for a react-query result. */
export function QueryBoundary<T>({
  query, skeleton, children,
}: { query: UseQueryResult<T, Error>; skeleton: ReactNode; children: (data: T) => ReactNode }) {
  if (query.isPending) return <>{skeleton}</>;
  if (query.isError) return <ErrorState error={query.error} onRetry={() => query.refetch()} />;
  return <>{children(query.data)}</>;
}
