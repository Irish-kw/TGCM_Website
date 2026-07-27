const numberFormatter = new Intl.NumberFormat('en-US');

function parseCsv(text) {
  const [headerLine, ...lines] = text.trim().split(/\r?\n/);
  const headers = headerLine.split(',');
  return lines.filter(Boolean).map((line) => {
    const values = line.split(',');
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function makeCell(tag, value) {
  const cell = document.createElement(tag);
  cell.textContent = value;
  return cell;
}

async function renderRunTable(container) {
  const source = container.dataset.csv;
  const scope = container.dataset.scope;
  try {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const rows = parseCsv(await response.text()).filter((row) => row.scope === scope);

    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrap';
    const table = document.createElement('table');
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    ['Run', 'Profile mix', '#Profiles', 'Total events', 'Size (GB)', 'Benign events', 'Malicious events', 'Malicious ratio (%)']
      .forEach((label) => headRow.appendChild(makeCell('th', label)));
    head.appendChild(headRow);
    table.appendChild(head);

    const body = document.createElement('tbody');
    rows.forEach((row) => {
      const tr = document.createElement('tr');
      tr.appendChild(makeCell('td', row.run));
      tr.appendChild(makeCell('td', row.profile_mix));
      tr.appendChild(makeCell('td', row.profiles));
      tr.appendChild(makeCell('td', numberFormatter.format(Number(row.total_events))));
      tr.appendChild(makeCell('td', Number(row.size_gb).toFixed(3)));
      tr.appendChild(makeCell('td', numberFormatter.format(Number(row.benign_events))));
      tr.appendChild(makeCell('td', numberFormatter.format(Number(row.malicious_events))));
      tr.appendChild(makeCell('td', Number(row.malicious_ratio_percent).toFixed(2)));
      body.appendChild(tr);
    });
    table.appendChild(body);
    wrapper.appendChild(table);
    container.replaceChildren(wrapper);
  } catch (error) {
    container.textContent = `Unable to load per-run statistics: ${error.message}`;
  }
}

document.querySelectorAll('[data-run-table]').forEach(renderRunTable);
