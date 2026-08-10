/**
 * HTML rendering utilities for the website
 */

export function renderPage(title: string, body: string): Response {
  const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${title} — ModelSwarm</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #e0e0e0; line-height: 1.6; }
    .container { max-width: 960px; margin: 0 auto; padding: 2rem; }
    header { border-bottom: 1px solid #1a1a2e; padding-bottom: 1rem; margin-bottom: 2rem; }
    header h1 { font-size: 1.5rem; color: #7c3aed; }
    header a { color: #7c3aed; text-decoration: none; }
    nav { margin-top: 0.5rem; }
    nav a { margin-right: 1rem; color: #888; font-size: 0.9rem; }
    nav a:hover { color: #7c3aed; }
    h2 { color: #a78bfa; margin: 1.5rem 0 1rem; }
    h3 { color: #c4b5fd; margin: 1rem 0 0.5rem; }
    .card { background: #12121a; border: 1px solid #1a1a2e; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; }
    .card h3 { margin-top: 0; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
    .badge-active { background: #064e3b; color: #6ee7b7; }
    .badge-queued { background: #1e3a5f; color: #93c5fd; }
    .badge-completed { background: #1e1b4b; color: #c4b5fd; }
    .badge-promoted { background: #4c1d95; color: #ddd6fe; }
    .metric { font-size: 2rem; font-weight: 700; color: #7c3aed; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
    a { color: #7c3aed; }
    a:hover { text-decoration: underline; }
    code { background: #1a1a2e; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.9rem; }
    pre { background: #1a1a2e; padding: 1rem; border-radius: 8px; overflow-x: auto; margin: 1rem 0; }
    pre code { background: none; padding: 0; }
    .btn { display: inline-block; background: #7c3aed; color: white; padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; }
    .btn:hover { background: #6d28d9; text-decoration: none; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th, td { text-align: left; padding: 0.5rem; border-bottom: 1px solid #1a1a2e; }
    th { color: #a78bfa; font-weight: 600; }
    footer { border-top: 1px solid #1a1a2e; margin-top: 2rem; padding-top: 1rem; color: #666; font-size: 0.85rem; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1><a href="/">ModelSwarm</a></h1>
      <nav>
        <a href="/">Home</a>
        <a href="/competitions">Competitions</a>
        <a href="/agents.md">agents.md</a>
      </nav>
    </header>
    <main>${body}</main>
    <footer>ModelSwarm — Autonomous Multi-Agent ML Research Platform</footer>
  </div>
</body>
</html>`;

  return new Response(html, {
    headers: { 'Content-Type': 'text/html; charset=utf-8' },
  });
}
