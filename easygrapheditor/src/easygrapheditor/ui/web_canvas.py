"""Shared HTML/SVG canvas and action protocol for the web hosts."""

from __future__ import annotations

import base64
import html
import io
import json
from typing import Any

from ..engine.nodes import NODE_REGISTRY, get_node
from .canvas import EditorState


def _preview_data_uri(value: Any) -> str:
    data = getattr(value, "data", None)
    if data is None:
        return ""
    try:
        import numpy as np
        from PIL import Image as PILImage

        array = np.asarray(data)
        if array.ndim == 2:
            image = PILImage.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8), mode="L")
        elif array.ndim == 3 and array.shape[-1] in (3, 4):
            image = PILImage.fromarray(array.astype(np.uint8), mode="RGBA" if array.shape[-1] == 4 else "RGB")
        else:
            return ""
        image.thumbnail((160, 120))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except (TypeError, ValueError):
        return ""


def _node_payload(state: EditorState, outputs: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Serialize node cards without leaking runtime-only engine objects."""
    rows: list[dict[str, Any]] = []
    report = state.run_report
    outputs = outputs or {}
    for nid, inst in state.graph.nodes.items():
        definition = get_node(inst.type_id)
        result = report.per_node.get(nid) if report else None
        working = state.working_nodes.get(nid, {})
        ports = lambda values: [
            {"key": port.key, "label": port.label or port.key, "dtype": port.dtype}
            for port in values
        ]
        rows.append({
            "id": nid, "title": definition.title if definition else inst.type_id,
            "x": inst.pos[0], "y": inst.pos[1], "collapsed": inst.collapsed,
            "selected": nid in state.selection, "inputs": ports(definition.inputs) if definition else [],
            "outputs": ports(definition.outputs) if definition else [], "params": dict(inst.params),
            "status": working.get("status", result.status if result else "not run"), "ms": round(result.ms, 1) if result else 0.0,
            "cached": bool(result and result.cache_hit),
            "stages": working.get("stages", list(result.stages) if result else []),
            "stage": working.get("stage", ""), "frac": working.get("frac", 0.0),
            "values": {key: repr(value)[:80] for key, value in outputs.get(nid, {}).items()},
            "previews": {key: uri for key, value in outputs.get(nid, {}).items()
                         if (uri := _preview_data_uri(value))},
        })
    return rows


def editor_canvas_html(
    state: EditorState,
    height: int = 540,
    *,
    include_script: bool = True,
    editable: bool = True,
    outputs: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Build the interactive, true-black SVG surface used by both web hosts."""
    payload = {
        "nodes": _node_payload(state, outputs),
        "links": [
            {"from": link.from_node, "from_port": link.from_port, "to": link.to_node, "to_port": link.to_port}
            for link in state.graph.links
        ],
        "viewport": {"x": state.viewport.x, "y": state.viewport.y, "zoom": state.viewport.zoom},
        "snap": state.snap_to_grid, "grid": state.grid_size, "settings": state.settings,
        "editable": editable,
        "library": [
            {"type_id": node.type_id, "title": node.title, "category": node.category}
            for node in sorted(NODE_REGISTRY.values(), key=lambda item: (item.category, item.title))
            if state.allowed_node_types is None or node.type_id in state.allowed_node_types
        ],
    }
    data = html.escape(json.dumps(payload), quote=True)
    instance = f"ege-{id(state):x}"
    editing_buttons = "" if not editable else (
        '<button data-action="live">live</button><button data-action="cache">forget cache</button>'
        '<button data-action="undo">undo</button><button data-action="redo">redo</button>'
        '<button data-action="add">add</button><button data-action="delete">delete</button>'
        '<button data-action="group">group</button><button data-action="ungroup">ungroup</button>'
        '<button data-action="collapse">collapse</button>'
        '<button data-action="uncollapse">expand</button><button data-action="snap">snap</button>'
        '<button data-action="param">param</button><button data-action="settings">settings</button>'
    )
    markup = f'''<div id="{instance}" class="ege-canvas" style="height:{int(height)}px" data-graph="{data}">
<style>
#{instance}.ege-canvas{{background:#000;color:#fff;border:1px solid #303030;min-height:260px;overflow:hidden;font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}
#{instance} .ege-bar{{height:30px;display:flex;align-items:center;gap:5px;padding:0 8px;border-bottom:1px solid #303030;overflow:auto}}
#{instance} button{{background:#000;color:#fff;border:1px solid #555;padding:2px 5px;cursor:pointer;white-space:nowrap;font:11px ui-monospace,SFMono-Regular,Menlo,monospace}}#{instance} button:hover{{border-color:#fff}}
#{instance} .ege-workspace{{height:calc(100% - 31px);display:flex;min-width:0}}#{instance} svg{{flex:1;min-width:0;height:100%;display:block;touch-action:none;background-image:radial-gradient(#252525 1px,transparent 1px);background-size:20px 20px}}
#{instance} .ege-dock{{width:240px;border-left:1px solid #303030;background:#0c0c0c;overflow:auto}}#{instance} .ege-tabs{{display:flex;border-bottom:1px solid #303030}}#{instance} .ege-tabs button{{flex:1;border:0;border-right:1px solid #303030;padding:7px 3px}}#{instance} .ege-dock-body{{padding:8px;line-height:1.45}}#{instance} .ege-dock-row{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px;padding:3px 0;border-bottom:1px solid #202020}}#{instance} .ege-dock-row span:last-child{{color:#aaa;text-align:right;overflow-wrap:anywhere}}#{instance} .ege-category{{color:#6ea8ff;margin-top:8px}}#{instance} .ege-library-item{{display:block;width:100%;text-align:left;border:0;padding:3px 0;color:#ddd}}
#{instance} .ege-port-preview{{display:block;width:100%;max-height:120px;object-fit:contain;margin:5px 0 9px;background:#000;border:1px solid #303030}}
#{instance} .link{{fill:none;stroke:#6ea8ff;stroke-width:2;cursor:pointer}}#{instance} .node{{fill:#090909;stroke:#777;stroke-width:1;cursor:grab}}#{instance} .node.selected{{stroke:#fff;stroke-width:2}}#{instance} .head{{fill:#151515}}#{instance} text{{fill:#fff;user-select:none;pointer-events:none}}#{instance} .port{{fill:#6ea8ff;stroke:#000;stroke-width:1;cursor:crosshair}}#{instance} .port.legal{{stroke:#fff;stroke-width:2}}#{instance} .meta{{fill:#aaa;font-size:10px}}#{instance} .marquee{{fill:#ffffff10;stroke:#fff;stroke-dasharray:3 2}}
</style>
<div class="ege-bar"><button data-action="run">run</button><button data-action="cancel">cancel</button>{editing_buttons}<button data-action="fit">fit</button><span class="status"></span></div>
<div class="ege-workspace"><svg role="application" aria-label="Node graph editor"><g class="world"></g></svg><aside class="ege-dock"><div class="ege-tabs"><button data-dock="library">Library</button><button data-dock="inspector">Inspector</button><button data-dock="types">Types</button></div><div class="ege-dock-body"></div></aside></div><script>(function(){{
const scope=typeof parentElement==='undefined'?document:parentElement,root=scope.querySelector({json.dumps('#' + instance)});if(!root||root.dataset.ready)return;root.dataset.ready='1';const g=JSON.parse(root.dataset.graph),svg=root.querySelector('svg'),world=root.querySelector('.world'),status=root.querySelector('.status'),dock=root.querySelector('.ege-dock-body');let dockTab='inspector';
const savedSettings=JSON.parse(localStorage.getItem('easygrapheditor.settings')||'{{}}');let view=g.viewport,pan=null,drag=null,wire=null,marquee=null;const nodeMap=new Map(g.nodes.map(n=>[n.id,n]));
const esc=s=>String(s).replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));const emit=(type,data={{}})=>{{const action={{type,data}};let pending;if(typeof trigger==='function')pending=trigger('click',action);if(typeof setTriggerValue==='function')pending=setTriggerValue('action',action);root.dispatchEvent(new CustomEvent('easygrapheditor:action',{{bubbles:true,detail:action}}));return pending;}};
const saveSettings=settings=>{{localStorage.setItem('easygrapheditor.settings',JSON.stringify(settings));emit('settings',{{settings}});}};if(Object.keys(savedSettings).length)emit('settings',{{settings:savedSettings}});
const point=e=>{{const r=svg.getBoundingClientRect();return {{x:e.clientX-r.left,y:e.clientY-r.top}};}};const local=p=>({{x:(p.x-view.x)/view.zoom,y:(p.y-view.y)/view.zoom}});
function tx(){{world.setAttribute('transform',`translate(${{view.x}} ${{view.y}}) scale(${{view.zoom}})`);status.textContent=`${{Math.round(view.zoom*100)}}% | ${{g.nodes.length}} nodes | ${{g.links.length}} links`;}}
const shortType=value=>String(value).replace(/^data[.]/,'');function nodeWidth(n){{let chars=n.title.length+18;const count=Math.max(n.inputs.length,n.outputs.length);for(let i=0;i<count;i++){{const a=n.inputs[i],b=n.outputs[i],left=a?`${{a.label}} : ${{shortType(a.dtype)}}`:'',right=b?`${{b.label}} : ${{shortType(b.dtype)}}`:'';chars=Math.max(chars,left.length+right.length+5);}}const params=Object.entries(n.params).slice(0,2).map(([k,v])=>`${{k}}=${{v}}`).join(' ');return Math.max(190,Math.min(360,Math.ceil(Math.max(chars,params.length+3)*6.4)));}}
function fit(){{if(!g.nodes.length)return;const r=svg.getBoundingClientRect(),minX=Math.min(...g.nodes.map(n=>n.x)),minY=Math.min(...g.nodes.map(n=>n.y)),maxX=Math.max(...g.nodes.map(n=>n.x+nodeWidth(n))),maxY=Math.max(...g.nodes.map(n=>n.y+Math.max(74,68+Math.max(n.inputs.length,n.outputs.length)*16)));const z=Math.max(.35,Math.min(2.5,(r.width-60)/Math.max(1,maxX-minX),(r.height-60)/Math.max(1,maxY-minY)));view={{x:30-minX*z,y:30-minY*z,zoom:z}};tx();}}
function dockRow(a,b){{return `<div class="ege-dock-row"><span>${{esc(a)}}</span><span>${{esc(b)}}</span></div>`;}}function renderDock(){{if(dockTab==='library'){{let last='',html='';for(const item of g.library){{if(item.category!==last){{last=item.category;html+=`<div class="ege-category">${{esc(last)}}</div>`;}}html+=`<button class="ege-library-item" data-add="${{esc(item.type_id)}}">${{esc(item.title)}}</button>`;}}dock.innerHTML=html;dock.querySelectorAll('[data-add]').forEach(button=>button.onclick=()=>emit('add',{{type_id:button.dataset.add,pos:[40,40]}}));return;}}if(dockTab==='types'){{const types=[...new Set(g.nodes.flatMap(n=>[...n.inputs,...n.outputs].map(p=>shortType(p.dtype))))].sort();dock.innerHTML=types.map(type=>dockRow(type,'port type')).join('')||'<span class="meta">No port types</span>';return;}}const n=g.nodes.find(item=>item.selected);if(!n){{dock.innerHTML='<span class="meta">Select a node</span>';return;}}let html=`<div>${{esc(n.title)}}</div>${{dockRow('Status',n.status)}}`;for(const p of n.inputs)html+=dockRow(p.label,`IN · ${{shortType(p.dtype)}}`);for(const p of n.outputs)html+=dockRow(p.label,`OUT · ${{shortType(p.dtype)}}`);for(const [key,value] of Object.entries(n.params))html+=dockRow(key,String(value));dock.innerHTML=html;}}
function portY(i){{return 44+i*16;}}function render(){{world.innerHTML='';for(const l of g.links){{const a=nodeMap.get(l.from),b=nodeMap.get(l.to);if(!a||!b)continue;const ai=Math.max(0,a.outputs.findIndex(p=>p.key===l.from_port)),bi=Math.max(0,b.inputs.findIndex(p=>p.key===l.to_port));const x1=a.x+nodeWidth(a),y1=a.y+portY(ai),x2=b.x,y2=b.y+portY(bi);world.insertAdjacentHTML('beforeend',`<path class="link" data-from="${{esc(l.from)}}" data-from-port="${{esc(l.from_port)}}" data-to="${{esc(l.to)}}" data-to-port="${{esc(l.to_port)}}" d="M${{x1}},${{y1}} C${{x1+70}},${{y1}} ${{x2-70}},${{y2}} ${{x2}},${{y2}}"/>`);}}
for(const n of g.nodes){{const w=nodeWidth(n),count=Math.max(n.inputs.length,n.outputs.length),h=n.collapsed?34:Math.max(74,34+count*16+34);let ports='';if(!n.collapsed){{n.inputs.forEach((p,i)=>ports+=`<circle class="port" data-dir="in" data-node="${{esc(n.id)}}" data-port="${{esc(p.key)}}" cx="0" cy="${{portY(i)}}" r="4"/><text class="meta" x="8" y="${{portY(i)+4}}">${{esc(p.label)}} : ${{esc(shortType(p.dtype))}}</text>`);n.outputs.forEach((p,i)=>ports+=`<circle class="port" data-dir="out" data-node="${{esc(n.id)}}" data-port="${{esc(p.key)}}" cx="${{w}}" cy="${{portY(i)}}" r="4"/><text class="meta" text-anchor="end" x="${{w-8}}" y="${{portY(i)+4}}">${{esc(p.label)}} : ${{esc(shortType(p.dtype))}}</text>`);}}const info=`${{esc(n.status)}} ${{n.ms}}ms${{n.cached?' cache':''}} s${{n.stages.length}}`,params=Object.entries(n.params).slice(0,2).map(([k,v])=>`${{k}}=${{v}}`).join(' ');world.insertAdjacentHTML('beforeend',`<g class="card" data-id="${{esc(n.id)}}" transform="translate(${{n.x}},${{n.y}})"><rect class="node ${{n.selected?'selected':''}}" width="${{w}}" height="${{h}}"/><rect class="head" width="${{w}}" height="26"/><text x="8" y="17">${{esc(n.title)}}</text><text class="meta" text-anchor="end" x="${{w-8}}" y="17">${{info}}</text>${{ports}}${{n.collapsed?'':`<text class="meta" x="8" y="${{h-8}}">${{esc(params)}}</text>`}}</g>`);}}tx();renderDock();}}render();root.querySelectorAll('[data-dock]').forEach(button=>button.onclick=()=>{{dockTab=button.dataset.dock;renderDock();}});if(view.x===0&&view.y===0&&view.zoom===1)requestAnimationFrame(fit);
root.querySelectorAll('button').forEach(b=>b.onclick=()=>{{const kind=b.dataset.action;if(kind==='fit'){{fit();}}else if(kind==='snap'){{for(const n of g.nodes){{n.x=Math.round(n.x/g.grid)*g.grid;n.y=Math.round(n.y/g.grid)*g.grid;}}render();emit('positions',{{positions:Object.fromEntries(g.nodes.map(n=>[n.id,[n.x,n.y]]))}});return;}}else if(kind==='add'){{const type_id=prompt('Node type','input.number');if(type_id)emit('add',{{type_id,pos:[0,0]}});return;}}else if(kind==='param'){{const n=g.nodes.find(n=>n.selected);const key=prompt('Parameter key');if(n&&key)emit('param',{{node:n.id,key,value:prompt('Value')}});return;}}else if(kind==='settings'){{const raw=prompt('Settings JSON',JSON.stringify(g.settings));try{{if(raw){{g.settings=JSON.parse(raw);saveSettings(g.settings);}}}}catch(error){{status.textContent=`settings error: ${{error.message}}`;}}return;}}else if(kind==='group')emit('group',{{nodes:g.nodes.filter(n=>n.selected).map(n=>n.id)}});else if(kind==='ungroup')emit('ungroup',{{}});else if(kind==='delete')emit('delete',{{nodes:g.nodes.filter(n=>n.selected).map(n=>n.id)}});else if(kind==='collapse'||kind==='uncollapse')emit('collapse',{{collapsed:kind==='collapse'}});else if(kind==='live'){{g.settings.live=!g.settings.live;saveSettings(g.settings);emit('live',{{value:g.settings.live}});return;}}emit(kind);}});
svg.addEventListener('pointerdown',e=>{{const port=e.target.closest('.port'),link=e.target.closest('.link'),card=e.target.closest('.card'),p=point(e);if(g.editable&&link){{wire={{from_node:link.dataset.from,from_port:link.dataset.fromPort,dir:'out'}};emit('disconnect',{{from_node:link.dataset.from,from_port:link.dataset.fromPort,to_node:link.dataset.to,to_port:link.dataset.toPort}});return;}}if(g.editable&&port){{wire={{from_node:port.dataset.node,from_port:port.dataset.port,dir:port.dataset.dir}};return;}}if(card){{const n=nodeMap.get(card.dataset.id);if(g.editable){{g.nodes.forEach(item=>item.selected=item===n);drag={{n,p,x:n.x,y:n.y}};render();svg.setPointerCapture(e.pointerId);emit('select',{{node:n.id}});}}}}else if(g.editable&&e.shiftKey){{marquee={{p,q:p}};}}else pan={{p,x:view.x,y:view.y}};}});
svg.addEventListener('pointermove',e=>{{const p=point(e);if(drag){{const q=local(p),start=local(drag.p);drag.n.x=drag.x+q.x-start.x;drag.n.y=drag.y+q.y-start.y;render();}}else if(pan){{view.x=pan.x+p.x-pan.p.x;view.y=pan.y+p.y-pan.p.y;tx();}}else if(marquee){{marquee.q=p;const a=local(marquee.p),b=local(p);world.querySelector('.marquee')?.remove();world.insertAdjacentHTML('beforeend',`<rect class="marquee" x="${{Math.min(a.x,b.x)}}" y="${{Math.min(a.y,b.y)}}" width="${{Math.abs(a.x-b.x)}}" height="${{Math.abs(a.y-b.y)}}"/>`);}}}});
svg.addEventListener('pointerup',e=>{{const port=e.target.closest('.port');if(wire&&port&&wire.dir==='out'&&port.dataset.dir==='in')emit('connect',{{from_node:wire.from_node,from_port:wire.from_port,to_node:port.dataset.node,to_port:port.dataset.port}});if(drag)emit('positions',{{positions:Object.fromEntries(g.nodes.map(n=>[n.id,[n.x,n.y]]))}});if(marquee){{const a=local(marquee.p),b=local(marquee.q),x0=Math.min(a.x,b.x),x1=Math.max(a.x,b.x),y0=Math.min(a.y,b.y),y1=Math.max(a.y,b.y);emit('marquee',{{nodes:g.nodes.filter(n=>n.x>=x0&&n.x<=x1&&n.y>=y0&&n.y<=y1).map(n=>n.id)}});}}if(pan)emit('viewport',view);drag=pan=wire=marquee=null;}});
svg.addEventListener('wheel',e=>{{e.preventDefault();const p=point(e),old=view.zoom;view.zoom=Math.max(.35,Math.min(2.5,old*(e.deltaY<0?1.1:1/1.1)));view.x=p.x-(p.x-view.x)*view.zoom/old;view.y=p.y-(p.y-view.y)*view.zoom/old;tx();emit('viewport',view);}},{{passive:false}});
const portPreview=(node,key)=>node?.previews?.[key]?`<img class="ege-port-preview" src="${{node.previews[key]}}" alt="${{esc(key)}} preview">`:'';
const renderInspectorDock=()=>{{const n=g.nodes.find(item=>item.selected);if(!n){{dock.innerHTML='<span class="meta">Select a node</span>';return;}}let markup=`<div>${{esc(n.title)}}</div>${{dockRow('Status',n.status)}}<div class="ege-category">Inputs</div>`;for(const p of n.inputs){{const link=g.links.find(item=>item.to===n.id&&item.to_port===p.key),source=link?nodeMap.get(link.from):null;markup+=dockRow(p.label,`IN · ${{shortType(p.dtype)}}`)+portPreview(source,link?.from_port);}}markup+='<div class="ege-category">Outputs</div>';for(const p of n.outputs)markup+=dockRow(p.label,`OUT · ${{shortType(p.dtype)}}`)+portPreview(n,p.key);markup+='<div class="ege-category">Parameters</div>';for(const [key,value] of Object.entries(n.params))markup+=dockRow(key,String(value));dock.innerHTML=markup;}};
root.querySelectorAll('[data-dock]').forEach(button=>button.onclick=()=>{{dockTab=button.dataset.dock;if(dockTab==='inspector')renderInspectorDock();else renderDock();}});
renderDock=(()=>{{const original=renderDock;return()=>{{if(dockTab==='inspector'){{renderInspectorDock();return;}}original();if(dockTab==='library')dock.querySelectorAll('[data-add]').forEach(button=>button.onclick=()=>{{const pending=emit('add',{{type_id:button.dataset.add,pos:[40,40]}}),reload=()=>location.reload();if(pending&&typeof pending.then==='function')pending.then(reload,reload);else setTimeout(reload,1800);}});}};}})();renderDock();
}})();</script></div>'''
    return markup if include_script else markup.split("<script>", 1)[0] + "</div>"


def editor_canvas_js(state: EditorState, *, editable: bool = True, outputs: dict[str, dict[str, Any]] | None = None) -> str:
    """Return the canvas initializer for hosts with separate JavaScript hooks."""
    markup = editor_canvas_html(state, editable=editable, outputs=outputs)
    return markup.split("<script>", 1)[1].rsplit("</script>", 1)[0]


def streamlit_canvas_js(state: EditorState, *, editable: bool = True, outputs: dict[str, dict[str, Any]] | None = None) -> str:
    """Streamlit v2 module source using its public trigger API."""
    return "export default function(component) { const { parentElement } = component; const setTriggerValue = (name, value) => component.setTriggerValue(name, value); " + editor_canvas_js(state, editable=editable, outputs=outputs) + " }"


__all__ = ["editor_canvas_html", "editor_canvas_js", "streamlit_canvas_js"]
