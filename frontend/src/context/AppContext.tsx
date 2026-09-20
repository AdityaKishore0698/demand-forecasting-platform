import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { useStores } from "../api/hooks";
import type { StoreInfo } from "../api/types";

interface AppCtx {
  storeId: number | null;
  store: StoreInfo | null;
  setStoreId: (id: number) => void;
  horizon: number;
  setHorizon: (h: number) => void;
  stores: StoreInfo[];
  storesLoading: boolean;
  storesError: Error | null;
  refetchStores: () => void;
}
const Ctx = createContext<AppCtx | null>(null);

const num = (v: string | null): number | null => {
  if (v == null) return null;
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : null;
};
const saved = (key: string) => { try { return localStorage.getItem(key); } catch { return null; } };
const params = new URLSearchParams(window.location.search);

export function AppProvider({ children }: { children: ReactNode }) {
  const { data, isLoading, error: storesError, refetch: refetchStores } = useStores();
  const stores = data?.stores ?? [];
  const [storeId, setStoreIdState] = useState<number | null>(num(params.get("store")) ?? num(saved("df-store")));
  const [horizon, setHorizonState] = useState<number>(num(params.get("horizon")) ?? num(saved("df-horizon")) ?? 14);

  // pick a valid default once the store list is known
  useEffect(() => {
    if (!stores.length) return;
    if (storeId == null || !stores.some((s) => s.store_id === storeId)) {
      setStoreIdState((stores.find((s) => s.store_id === 1) ?? stores[0]).store_id);
    }
  }, [stores, storeId]);

  const setStoreId = useCallback((id: number) => {
    setStoreIdState(id);
    try { localStorage.setItem("df-store", String(id)); } catch { /* ignore */ }
  }, []);
  const setHorizon = useCallback((h: number) => {
    setHorizonState(h);
    try { localStorage.setItem("df-horizon", String(h)); } catch { /* ignore */ }
  }, []);

  const store = useMemo(() => stores.find((s) => s.store_id === storeId) ?? null, [stores, storeId]);
  const value = useMemo(
    () => ({ storeId, store, setStoreId, horizon, setHorizon, stores, storesLoading: isLoading, storesError: storesError ?? null, refetchStores: () => { void refetchStores(); } }),
    [storeId, store, setStoreId, horizon, setHorizon, stores, isLoading, storesError, refetchStores],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp() {
  const v = useContext(Ctx);
  if (!v) throw new Error("useApp must be used inside AppProvider");
  return v;
}
