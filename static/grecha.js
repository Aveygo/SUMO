const LOREM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.";

async function tag(name, ...children) {
    const result = document.createElement(name);

    for (const child of children) {
        if (child.constructor==Object) {
            Object.keys(child).forEach(key => result.setAttribute(key, child[key]));
        } else if (typeof(child) === 'string') {
            result.innerHTML += child;
        } else {
            new Promise(async () => {    
                result.appendChild(await child);
            });
        }
    }

    return result;
}

const MUNDANE_TAGS = ["canvas", "h1", "h2", "h3", "p", "a", "div", "span", "select", "img", "input", "video", "source"];
for (let tagName of MUNDANE_TAGS) {
    window[tagName] = (...children) => tag(tagName, ...children);
}

function router(routes) {
    let result = div();

    async function syncHash() {
        let hashLocation = document.location.hash.split('#')[1];
        
        if (!hashLocation) {
            hashLocation = '/';
        }

        if (hashLocation.includes("?")) {
            hashLocation = hashLocation.split("?")[0]
        }

        if (!(hashLocation in routes)) {
            // TODO(#2): make the route404 customizable in the router component
            const route404 = '/404';

            console.assert(route404 in routes);
            hashLocation = route404;
        }

        (await result).replaceChildren(await routes[hashLocation]());

        return result;
    };

    syncHash();
    result.refresh = syncHash;

    // TODO(#3): there is way to "destroy" an instance of the router to make it remove it's "hashchange" callback
    window.addEventListener("hashchange", syncHash);
    return result;
}