import { createFileRoute, Link } from "@tanstack/react-router";
import { ShieldCheck, FileSearch, GaugeCircle, MessagesSquare } from "lucide-react";
//import logo from "@/public/ulak-logo-beyaz-2.png";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "ULAK Quality Test Analyst — AI Test Analysis Workspace" },
      {
        name: "description",
        content:
          "Secure AI assistant for ULAK Haberleşme quality engineers: upload test reports and logs, get defect analysis, and keep every conversation on record.",
      },
      { property: "og:title", content: "ULAK Quality Test Analyst" },
      {
        property: "og:description",
        content:
          "Secure AI assistant for ULAK Haberleşme quality engineers: test report analysis, defect triage and conversation history.",
      },
    ],
  }),
  component: Landing,
});

const features = [
  {
    icon: FileSearch,
    title: "Report & log analysis",
    body: "Upload test reports, logs or CSV result sets and get structured findings in seconds.",
  },
  {
    icon: GaugeCircle,
    title: "Defect triage",
    body: "Severity assessment, root-cause hypotheses and regression risk for every finding.",
  },
  {
    icon: MessagesSquare,
    title: "Full conversation history",
    body: "Every analysis thread is stored against your account and searchable later.",
  },
  {
    icon: ShieldCheck,
    title: "Account-scoped access",
    body: "Files and chats are private to the engineer who uploaded them.",
  },
];

function Landing() {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-20 border-b border-border bg-background/85 backdrop-blur">
        <div className="mx-auto flex h-[88px] max-w-[1400px] items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <img src="/ulak-logo-beyaz-2.png" alt="ULAK agent mark" width={90} height={40} className="h-10 w-auto" />
            <div className="leading-tight">
              <p className="text-lg font-extrabold tracking-tight">ULAK</p>
              <p className="text-xs text-muted-foreground">Quality Test Analyst</p>
            </div>
          </div>
          <Link
            to="/auth"
            className="rounded-full bg-primary px-7 py-3 text-sm font-bold text-primary-foreground transition-all duration-200 hover:bg-primary-light hover:glow-primary"
          >
            Sign in
          </Link>
        </div>
      </header>

      <main>
        <section className="relative overflow-hidden">
          <div className="absolute inset-0 grid-backdrop opacity-60" aria-hidden="true" />
          <div className="absolute inset-0 bg-hero-gradient" aria-hidden="true" />
          <div className="relative mx-auto max-w-[1400px] px-6 py-28 lg:py-36">
            <span className="inline-flex items-center gap-2 rounded-full border border-primary/40 bg-primary/10 px-4 py-1.5 text-xs font-semibold uppercase tracking-[0.18em] text-primary">
              ULAK Haberleşme · Internal
            </span>
            <h1 className="mt-8 max-w-4xl text-5xl font-extrabold leading-[1.05] tracking-tight lg:text-7xl">
              Your AI <span className="text-primary">quality test analyst</span> for telecom
              programs
            </h1>
            <p className="mt-6 max-w-2xl text-lg leading-relaxed text-muted-foreground lg:text-xl">
              Ask questions about test campaigns, upload evidence files, and get analyst-grade
              findings — defect severity, coverage gaps and release risk — in one secure workspace.
            </p>
            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/auth"
                className="rounded-full bg-primary px-9 py-4 text-base font-bold text-primary-foreground transition-all duration-200 hover:bg-primary-light hover:glow-primary"
              >
                Start analyzing
              </Link>
              <a
                href="#capabilities"
                className="rounded-full border-2 border-primary px-9 py-4 text-base font-bold text-foreground transition-colors duration-200 hover:bg-primary/10"
              >
                See capabilities
              </a>
            </div>
          </div>
        </section>

        <section id="capabilities" className="mx-auto max-w-[1400px] px-6 pb-28">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {features.map((f) => (
              <article key={f.title} className="panel bg-card-gradient p-8 transition-transform duration-200 hover:scale-[1.02]">
                <f.icon className="h-7 w-7 text-primary" strokeWidth={2} />
                <h2 className="mt-6 text-xl font-bold">{f.title}</h2>
                <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{f.body}</p>
              </article>
            ))}
          </div>
        </section>
      </main>

      <footer className="border-t border-border py-10">
        <div className="mx-auto max-w-[1400px] px-6 text-sm text-muted-foreground">
          © {new Date().getFullYear()} ULAK Haberleşme — internal quality engineering tooling.
        </div>
      </footer>
    </div>
  );
}
