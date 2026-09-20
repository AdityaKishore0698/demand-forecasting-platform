import type { ReactNode } from "react";
import { InfoTip } from "./InfoTip";

interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  info?: string;
  flush?: boolean;
  hover?: boolean;
  className?: string;
  children?: ReactNode;
}

export function Card({ title, subtitle, actions, info, flush, hover, className = "", children }: CardProps) {
  return (
    <section className={`card ${hover ? "card--hover" : ""} ${className}`}>
      {(title || actions) && (
        <header className="card__head">
          <div>
            <div className="card__title">
              <h2>{title}</h2>
              {info && <InfoTip text={info} />}
            </div>
            {subtitle && <p className="card__sub">{subtitle}</p>}
          </div>
          {actions && <div className="card__actions">{actions}</div>}
        </header>
      )}
      <div className={flush ? "card__body card__body--flush" : "card__body"}>{children}</div>
    </section>
  );
}
